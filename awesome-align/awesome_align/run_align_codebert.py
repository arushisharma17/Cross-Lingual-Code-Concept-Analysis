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
import awesome_align.modeling as modeling
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM, PreTrainedTokenizer, PreTrainedModel

def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path, max_length=2000, offsets=None):
        assert os.path.isfile(file_path)
        print('Loading the dataset...')
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.offsets = offsets
        self.max_length = max_length
        self.is_roberta = isinstance(tokenizer, RobertaTokenizerFast)

    def process_line(self, worker_id, line):
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None
        
        src, tgt = line.split(' ||| ')
        if src.rstrip() == '' or tgt.rstrip() == '':
            return None
    
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        
        # Handle RoBERTa tokenization differently
        if self.is_roberta:
            # Direct tokenization for RoBERTa
            encoded_src = self.tokenizer(
                sent_src,
                is_split_into_words=True,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            encoded_tgt = self.tokenizer(
                sent_tgt,
                is_split_into_words=True,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            ids_src = encoded_src['input_ids']
            ids_tgt = encoded_tgt['input_ids']
            
            # Get word IDs for alignment
            bpe2word_map_src = encoded_src.word_ids()
            bpe2word_map_tgt = encoded_tgt.word_ids()
            
            # Filter out None values and adjust indexing
            bpe2word_map_src = [i if i is not None else -1 for i in bpe2word_map_src]
            bpe2word_map_tgt = [i if i is not None else -1 for i in bpe2word_map_tgt]

        if ids_src.shape[1] <= 2 or ids_tgt.shape[1] <= 2:
            return None

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
                    empty_tensor = torch.tensor([self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
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


def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    def collate(examples):
        worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

    offsets = find_offsets(args.data_file, args.num_workers)
    dataset = LineByLineTextDataset(
        tokenizer, 
        file_path=args.data_file, 
        max_length=args.max_length,
        offsets=offsets
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=args.num_workers)

    if args.output_file is not None:
        outf = open(args.output_file, 'w', encoding='utf-8')
        writers = [outf]  # Initialize writers list with the output file

    model.eval()
    for batch in tqdm(dataloader, desc="Extracting"):
        with torch.no_grad():
            worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
            word_aligns = model.get_aligned_word(
                ids_src, ids_tgt,
                bpe2word_map_src[0], bpe2word_map_tgt[0],
                args.device,
                align_layer=args.align_layer,
                extraction=args.extraction,
                softmax_threshold=args.softmax_threshold,
                test=True,
                output_prob=args.output_prob_file is not None
            )
            # Collect all alignments for the current sentence
            alignments = [f"{src_idx}-{tgt_idx}" for src_idx, tgt_idx, _ in word_aligns]
            # Remove duplicate alignments
            unique_alignments = list(set(alignments))
            # Write all unique alignments for the current sentence on a single line
            writers[0].write(" ".join(unique_alignments) + "\n")

    # Ensure writers is defined before using it
    if 'writers' in locals():
        merge_files(writers)

    if args.output_file is not None:
        outf.close()
        print(f"\nSuccessfully written alignments to {args.output_file}")


def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--data_file", type=str, required=True, help="The input data file (a parallel corpus)."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="The output file (word alignments)."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        choices=["bert", "codebert"],
        help="Type of model to use (bert or codebert)",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
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
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
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
    parser.add_argument(
        "--max_length",
        default=2048,
        type=int,
        help="Maximum sequence length"
    )

    args = parser.parse_args()
    
    # Handle model type specific configurations
    if "roberta" in args.model_name_or_path or "codebert" in args.model_name_or_path:
        config_class = RobertaConfig
        model_class = RobertaForMaskedLM
        tokenizer_class = RobertaTokenizerFast
        
        if args.model_name_or_path is None:
            args.model_name_or_path = "microsoft/codebert-base"
            
        tokenizer = RobertaTokenizerFast.from_pretrained(
            args.model_name_or_path,
            do_lower_case=False,
            add_prefix_space=True,
            cache_dir=args.cache_dir
        )
        
        model = RobertaForMaskedLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir
        )
        
        # Add BERT's get_aligned_word method to RobertaForMaskedLM
        from awesome_align.modeling import BertForMaskedLM
        RobertaForMaskedLM.get_aligned_word = BertForMaskedLM.get_aligned_word
        model.get_aligned_word = RobertaForMaskedLM.get_aligned_word.__get__(model, RobertaForMaskedLM)
        
        # Update model constants for RoBERTa/CodeBERT
        modeling.PAD_ID = tokenizer.pad_token_id
        modeling.CLS_ID = tokenizer.bos_token_id  # RoBERTa uses <s> instead of [CLS]
        modeling.SEP_ID = tokenizer.eos_token_id  # RoBERTa uses </s> instead of [SEP]

    # Update model constants
    modeling.PAD_ID = tokenizer.pad_token_id
    modeling.CLS_ID = tokenizer.cls_token_id
    modeling.SEP_ID = tokenizer.sep_token_id

    # Update cache handling
    if args.cache_dir is None:
        args.cache_dir = os.path.join("cache", args.model_type)
    os.makedirs(args.cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    model.to(device)

    word_align(args, model, tokenizer)

if __name__ == "__main__":
    main()
