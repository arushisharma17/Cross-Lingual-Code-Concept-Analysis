# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import random
import itertools
import os
import shutil
import tempfile

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, T5ForConditionalGeneration


def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer: AutoTokenizer, file_path, offsets=None):
        assert os.path.isfile(file_path)
        print('Loading the dataset...')
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.offsets = offsets

    def process_line(self, worker_id, line):
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None
        
        src, tgt = line.split(' ||| ')
        if src.rstrip() == '' or tgt.rstrip() == '':
            return None
    
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        
        # Match training code's tokenization approach
        src_encoding = self.tokenizer(
            src.strip(), 
            return_tensors='pt', 
            truncation=True, 
            max_length=self.tokenizer.model_max_length
        )
        tgt_encoding = self.tokenizer(
            tgt.strip(), 
            return_tensors='pt', 
            truncation=True, 
            max_length=self.tokenizer.model_max_length
        )
        
        ids_src = src_encoding['input_ids']
        ids_tgt = tgt_encoding['input_ids']

        if ids_src.numel() == 0 or ids_tgt.numel() == 0:
            return None

        # Word-to-BPE mapping for RoBERTa tokenizer
        bpe2word_map_src = []
        curr_word = 0
        for token in self.tokenizer.convert_ids_to_tokens(ids_src[0]):
            if token.startswith('Ġ'):
                curr_word += 1
            bpe2word_map_src.append(curr_word - 1)

        bpe2word_map_tgt = []
        curr_word = 0
        for token in self.tokenizer.convert_ids_to_tokens(ids_tgt[0]):
            if token.startswith('Ġ'):
                curr_word += 1
            bpe2word_map_tgt.append(curr_word - 1)

        return (worker_id, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt) 

    def __iter__(self):
        if self.offsets is not None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id
            offset_start = self.offsets[worker_id]
            offset_end = self.offsets[worker_id+1] if worker_id+1 < len(self.offsets) else None
        else:
            offset_start = 0
            offset_end = None
            worker_id = 0

        with open(self.file_path, encoding="utf-8") as f:
            f.seek(offset_start)
            line = f.readline()
            while line:
                processed = self.process_line(worker_id, line)
                if processed is None:
                    print(f'Line "{line.strip()}" (offset in bytes: {f.tell()}) is not in the correct format. Skipping...')
                    empty_tensor = torch.tensor([self.tokenizer.bos_token_id, 999, self.tokenizer.eos_token_id])
                    empty_sent = ''
                    yield (worker_id, empty_tensor, empty_tensor, [-1], [-1], empty_sent, empty_sent)
                else:



                    yield processed
                if offset_end is not None and f.tell() >= offset_end:
                    break
                line = f.readline()

def find_offsets(filename, num_workers):
    if num_workers <= 1:
        return None
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_workers
        offsets = [0]
        for i in range(1, num_workers):
            f.seek(chunk_size * i)
            pos = f.tell()
            while True:
                try:
                    l=f.readline()
                    break
                except UnicodeDecodeError:
                    pos -= 1
                    f.seek(pos)
            offsets.append(f.tell())
    return offsets

def open_writer_list(filename, num_workers):
    writer = open(filename, 'w+', encoding='utf-8')
    writers = [writer]
    if num_workers > 1:
        writers.extend([tempfile.TemporaryFile(mode='w+', encoding='utf-8') for i in range(1, num_workers)])
    return writers

def merge_files(writers):
    if len(writers) == 1:
        writers[0].close()
        return

    for i, writer in enumerate(writers[1:], 1):
        writer.seek(0)
        shutil.copyfileobj(writer, writers[0])
        writer.close()
    writers[0].close()
    return


def word_align(args, model: T5ForConditionalGeneration, tokenizer: AutoTokenizer):
    def collate(examples):
        worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

    offsets = find_offsets(args.data_file, args.num_workers)
    dataset = LineByLineTextDataset(tokenizer, file_path=args.data_file, offsets=offsets)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
    )

    model.to(args.device)
    model.eval()
    tqdm_iterator = trange(0, desc="Extracting")

    writers = open_writer_list(args.output_file, args.num_workers) 
    if args.output_prob_file is not None:
        prob_writers = open_writer_list(args.output_prob_file, args.num_workers)
    if args.output_word_file is not None:
        word_writers = open_writer_list(args.output_word_file, args.num_workers)

    for batch in dataloader:
        with torch.no_grad():
            worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
            word_aligns_list = model.get_aligned_word(
                ids_src=ids_src, 
                ids_tgt=ids_tgt, 
                bpe2word_map_src=bpe2word_map_src, 
                bpe2word_map_tgt=bpe2word_map_tgt, 
                device=args.device,
                align_layer=args.align_layer,
                extraction=args.extraction,
                softmax_threshold=args.softmax_threshold,
                test=True,
                output_prob=(args.output_prob_file is not None)
            )
            for worker_id, word_aligns, sent_src, sent_tgt in zip(worker_ids, word_aligns_list, sents_src, sents_tgt):
                output_str = []
                if args.output_prob_file is not None:
                    output_prob_str = []
                if args.output_word_file is not None:
                    output_word_str = []
                for word_align in word_aligns:
                    if word_align[0] != -1:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                        if args.output_prob_file is not None:
                            output_prob_str.append(f'{word_aligns[word_align]}')
                        if args.output_word_file is not None:
                            output_word_str.append(f'{sent_src[word_align[0]]}<sep>{sent_tgt[word_align[1]]}')
                writers[worker_id].write(' '.join(output_str)+'\n')
                if args.output_prob_file is not None:
                    prob_writers[worker_id].write(' '.join(output_prob_str)+'\n')
                if args.output_word_file is not None:
                    word_writers[worker_id].write(' '.join(output_word_str)+'\n')
            tqdm_iterator.update(len(ids_src))

    merge_files(writers)
    if args.output_prob_file is not None:
        merge_files(prob_writers)
    if args.output_word_file is not None:
        merge_files(word_writers)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_file", default=None, type=str, required=True, help="The input data file (a text file)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output file."
    )
    parser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, help='softmax or entmax15'
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.001
    )
    parser.add_argument(
        "--output_prob_file", default=None, type=str, help='The output probability file.'
    )
    parser.add_argument(
        "--output_word_file", default=None, type=str, help='The output word file.'
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Set seed
    set_seed(args)

    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You must specify a CodeRosetta model path using --model_name_or_path"
        )

    word_align(args, model, tokenizer)

if __name__ == "__main__":
    main()
def get_cross_attention_alignments(model, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, align_layer=-1):
    """Get cross-attention alignments from T5 model"""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        output_attentions=True,
    )
    
    # Get cross-attention matrices from T5 outputs
    cross_attentions = outputs.cross_attentions
    if align_layer < 0:
        align_layer = len(cross_attentions) + align_layer
    attention_matrix = cross_attentions[align_layer]
    # Average over attention heads
    attention_matrix = attention_matrix.mean(dim=1)
    return attention_matrix

def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """Shift decoder input ids right for T5."""
    # T5 handles this internally, but keeping function for compatibility
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

def get_aligned_word(self, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, device, 
                    align_layer=-1, extraction='softmax', softmax_threshold=0.001, test=False, output_prob=False):
    """Get word alignments using T5 cross-attention"""
    batch_size = ids_src.size(0)
    results = []
    
    for idx in range(batch_size):
        src_ids = ids_src[idx:idx+1].to(device)
        tgt_ids = ids_tgt[idx:idx+1].to(device)
        src_mask = (src_ids != self.config.pad_token_id).float()
        tgt_mask = (tgt_ids != self.config.pad_token_id).float()
        
        # Generate decoder inputs
        decoder_input_ids = shift_tokens_right(
            tgt_ids,
            self.config.pad_token_id,
            self.config.decoder_start_token_id
        )
        
        # Get attention matrix
        attention_matrix = get_cross_attention_alignments(
            self,
            src_ids,
            src_mask,
            decoder_input_ids,
            tgt_mask,
            align_layer
        )
        
        attention_matrix = attention_matrix[:, :tgt_mask.sum().int(), :src_mask.sum().int()]
        
        src_len = len(bpe2word_map_src[idx])
        tgt_len = len(bpe2word_map_tgt[idx])
        word_matrix = torch.zeros((src_len, tgt_len), device=device)
        
        # Aggregate BPE-level attention to word-level
        for i in range(attention_matrix.size(1)):
            for j in range(attention_matrix.size(2)):
                tgt_idx = bpe2word_map_tgt[idx][i] if i < len(bpe2word_map_tgt[idx]) else -1
                src_idx = bpe2word_map_src[idx][j] if j < len(bpe2word_map_src[idx]) else -1
                if tgt_idx >= 0 and src_idx >= 0:
                    word_matrix[src_idx, tgt_idx] += attention_matrix[0, i, j]
        
        # Normalize word-level attention scores
        word_matrix = word_matrix / (word_matrix.sum(dim=-1, keepdim=True) + 1e-12)
        
        # For each source token, find max probability target token
        max_attentions, max_indices = word_matrix.max(dim=1)
        alignment_dict = {}
        
        # Only keep alignments above threshold
        for src_idx in range(src_len):
            max_prob = max_attentions[src_idx].item()
            if max_prob > softmax_threshold:
                tgt_idx = max_indices[src_idx].item()
                if output_prob:
                    alignment_dict[(src_idx, tgt_idx)] = max_prob
                else:
                    alignment_dict[(src_idx, tgt_idx)] = 1
        
        results.append(alignment_dict)
    
    return results

# Add the method to T5ForConditionalGeneration instead of EncoderDecoderModel
T5ForConditionalGeneration.get_aligned_word = get_aligned_word

