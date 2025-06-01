# awesome_align/awesome_align/run_align_codebert.py
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
from transformers import AutoModel, AutoTokenizer, AutoConfig

from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer
from awesome_align.modeling_utils import PreTrainedModel


def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path, offsets=None):
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
    
        # Split the sentences into words
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        
        # Check if we're using a RoBERTa-based tokenizer
        is_roberta = any(name in self.tokenizer.__class__.__name__.lower() for name in ["roberta", "codebert"]) 
        
        # Tokenize each word
        token_src = [self.tokenizer.tokenize(word) for word in sent_src]
        token_tgt = [self.tokenizer.tokenize(word) for word in sent_tgt]
        
        # Convert tokens to ids
        wid_src = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        
        # Flatten the token ids
        flat_ids_src = list(itertools.chain(*wid_src))
        flat_ids_tgt = list(itertools.chain(*wid_tgt))
        
        # Handle special tokens based on tokenizer type
        if is_roberta:
            # RoBERTa/CodeBERT uses <s> and </s>
            cls_token_id = self.tokenizer.cls_token_id if hasattr(self.tokenizer, 'cls_token_id') else self.tokenizer.bos_token_id
            sep_token_id = self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else self.tokenizer.eos_token_id
        else:
            # BERT uses [CLS] and [SEP]
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
        
        # Create input sequences with special tokens
        ids_src = torch.tensor([cls_token_id] + flat_ids_src + [sep_token_id])
        ids_tgt = torch.tensor([cls_token_id] + flat_ids_tgt + [sep_token_id])
        
        # Check if we have enough content
        if len(ids_src) <= 2 or len(ids_tgt) <= 2:
            return None
        
        # Create token to word mapping
        bpe2word_map_src = []
        for i, word_tokens in enumerate(token_src):
            bpe2word_map_src += [i for _ in word_tokens]
        
        bpe2word_map_tgt = []
        for i, word_tokens in enumerate(token_tgt):
            bpe2word_map_tgt += [i for _ in word_tokens]
        
        # Add mapping for special tokens
        bpe2word_map_src = [-1] + bpe2word_map_src + [-1]  # For CLS and SEP tokens
        bpe2word_map_tgt = [-1] + bpe2word_map_tgt + [-1]  # For CLS and SEP tokens
        
        return (worker_id, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt) 

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
                    cls_id = self.tokenizer.cls_token_id if hasattr(self.tokenizer, 'cls_token_id') else self.tokenizer.bos_token_id
                    sep_id = self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else self.tokenizer.eos_token_id
                    empty_tensor = torch.tensor([cls_id, 999, sep_id])
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

# Wrapper class to adapt CodeBERT to work with awesome-align
class CodeBERTForWordAlignment(torch.nn.Module):
    def __init__(self, model_name_or_path, tokenizer, config=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.tokenizer = tokenizer
        # Check if we're using a RoBERTa-based model
        self.is_roberta = any(name in model_name_or_path.lower() for name in ["roberta", "codebert"])
        
        # For RoBERTa models, extract the components we need to use directly
        if self.is_roberta:
            # Get embeddings components directly
            self.word_embeddings = self.model.embeddings.word_embeddings
            self.position_embeddings = self.model.embeddings.position_embeddings
            self.token_type_embeddings = self.model.embeddings.token_type_embeddings
            self.LayerNorm = self.model.embeddings.LayerNorm
            self.dropout = self.model.embeddings.dropout
            # Get encoder
            self.encoder = self.model.encoder
    
    def custom_embeddings(self, input_ids):
        """Custom embedding function to bypass the token_type_ids issue in RoBERTa"""
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device
        
        # RoBERTa has a position embedding limit (typically 512 or 514)
        # Get the maximum position embedding size
        max_position = self.position_embeddings.weight.size(0)
        
        # Create position IDs with clamping to the maximum position
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        # Clamp position IDs to the maximum position embedding size
        position_ids = torch.clamp(position_ids, 0, max_position - 1)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        
        # Create token type IDs (all zeros for RoBERTa)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # Get embeddings
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_aligned_word(self, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, device, src_len, tgt_len, 
                        align_layer=8, extraction='softmax', softmax_threshold=0.001, test=False, output_prob=False, word_aligns=None):
        # Move inputs to the correct device
        ids_src = ids_src.to(device)
        ids_tgt = ids_tgt.to(device)
        
        # Create attention masks
        attention_mask_src = torch.ones_like(ids_src)
        attention_mask_src[ids_src == self.tokenizer.pad_token_id] = 0
        attention_mask_tgt = torch.ones_like(ids_tgt)
        attention_mask_tgt[ids_tgt == self.tokenizer.pad_token_id] = 0

        # Get batch size and sequence lengths
        batch_size, src_seq_len = ids_src.shape
        _, tgt_seq_len = ids_tgt.shape
        
        # For RoBERTa-based models, we need to handle the forward pass differently
        with torch.no_grad():
            if self.is_roberta:
                # Use our custom embedding function
                embedding_output_src = self.custom_embeddings(ids_src)
                embedding_output_tgt = self.custom_embeddings(ids_tgt)
                
                # Extended attention mask for src and tgt
                extended_attention_mask_src = attention_mask_src.unsqueeze(1).unsqueeze(2)
                extended_attention_mask_src = extended_attention_mask_src.to(dtype=next(self.model.parameters()).dtype)
                extended_attention_mask_src = (1.0 - extended_attention_mask_src) * -10000.0
                
                extended_attention_mask_tgt = attention_mask_tgt.unsqueeze(1).unsqueeze(2)
                extended_attention_mask_tgt = extended_attention_mask_tgt.to(dtype=next(self.model.parameters()).dtype)
                extended_attention_mask_tgt = (1.0 - extended_attention_mask_tgt) * -10000.0
                
                # Pass through encoder
                encoder_outputs_src = self.encoder(
                    embedding_output_src,
                    attention_mask=extended_attention_mask_src,
                    output_hidden_states=True
                )
                encoder_outputs_tgt = self.encoder(
                    embedding_output_tgt,
                    attention_mask=extended_attention_mask_tgt,
                    output_hidden_states=True
                )
                
                # Get hidden states
                hidden_states_src = encoder_outputs_src.hidden_states
                hidden_states_tgt = encoder_outputs_tgt.hidden_states
            else:
                # For BERT models, use the standard forward pass
                outputs_src = self.model(
                    input_ids=ids_src,
                    attention_mask=attention_mask_src,
                    output_hidden_states=True
                )
                outputs_tgt = self.model(
                    input_ids=ids_tgt,
                    attention_mask=attention_mask_tgt,
                    output_hidden_states=True
                )
                hidden_states_src = outputs_src.hidden_states
                hidden_states_tgt = outputs_tgt.hidden_states
        
        # Use the specified layer's hidden states
        # Ensure align_layer is in the valid range
        num_layers = len(hidden_states_src)
        align_layer = min(align_layer, num_layers - 1)
        align_layer = max(0, align_layer)  # Ensure it's at least 0
        
        hidden_states_src_layer = hidden_states_src[align_layer]
        hidden_states_tgt_layer = hidden_states_tgt[align_layer]

        # Perform word alignment
        attention_probs = torch.bmm(
            hidden_states_src_layer, 
            hidden_states_tgt_layer.transpose(1, 2)
        )
        
        # Handle different extraction methods
        if extraction == 'softmax':
            attention_probs = torch.softmax(attention_probs, dim=-1)
            # Apply softmax threshold to avoid all alignments
            if softmax_threshold > 0:
                attention_probs[attention_probs < softmax_threshold] = 0
        elif extraction == 'entmax15':
            # Use sparsemax extraction from awesome_align
            from awesome_align.sparsemax import entmax15
            attention_probs = entmax15(attention_probs, dim=-1)
        
        word_aligns = []
        
        # Get special token IDs based on tokenizer type
        if self.is_roberta:
            # RoBERTa/CodeBERT special tokens
            cls_token_id = self.tokenizer.cls_token_id if hasattr(self.tokenizer, 'cls_token_id') else self.tokenizer.bos_token_id
            sep_token_id = self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else self.tokenizer.eos_token_id
        else:
            # BERT special tokens
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            
        pad_token_id = self.tokenizer.pad_token_id
        special_tokens = [cls_token_id, sep_token_id, pad_token_id]
        
        # Convert subword-level alignments to word-level alignments
        for i in range(batch_size):
            align_dict = {}
            
            # Use actual sequence lengths from the input tensors
            src_seq_len = ids_src.size(1)
            tgt_seq_len = ids_tgt.size(1)
            
            # Make sure bpe2word_map indices don't exceed sequence lengths
            src_mapping_len = min(len(bpe2word_map_src[i]), src_seq_len)
            tgt_mapping_len = min(len(bpe2word_map_tgt[i]), tgt_seq_len)
            
            # Process each source token
            for j_src in range(src_mapping_len):
                if j_src >= len(ids_src[i]) or ids_src[i][j_src].item() in special_tokens:
                    continue
                    
                if bpe2word_map_src[i][j_src] != -1:
                    src_word_idx = bpe2word_map_src[i][j_src]
                    max_prob = 0
                    max_idx = -1
                    
                    # Find the best target token alignment
                    for j_tgt in range(tgt_mapping_len):
                        if j_tgt >= len(ids_tgt[i]) or ids_tgt[i][j_tgt].item() in special_tokens:
                            continue
                            
                        if bpe2word_map_tgt[i][j_tgt] != -1:
                            prob = attention_probs[i, j_src, j_tgt].item()
                            if prob > max_prob:
                                max_prob = prob
                                max_idx = bpe2word_map_tgt[i][j_tgt]
                    
                    if max_idx != -1:
                        if (src_word_idx, max_idx) not in align_dict or align_dict[(src_word_idx, max_idx)] < max_prob:
                            align_dict[(src_word_idx, max_idx)] = max_prob
            
            cur_aligns = {}
            for (src_idx, tgt_idx), prob in align_dict.items():
                if (src_idx, tgt_idx) not in cur_aligns or prob > cur_aligns[(src_idx, tgt_idx)]:
                    cur_aligns[(src_idx, tgt_idx)] = prob
            
            word_aligns.append(cur_aligns)
        
        return word_aligns

# Move collate function outside of word_align function
def collate_fn(examples):
    worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
    ids_src = pad_sequence(ids_src, batch_first=True, padding_value=0)  # Will update padding value later
    ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=0)  # Will update padding value later
    return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

def word_align(args, model, tokenizer):
    # Update the padding value with tokenizer's pad_token_id
    global collate_fn
    original_collate = collate_fn
    def collate_with_tokenizer(examples):
        worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = original_collate(examples)
        ids_src = pad_sequence([ids for ids in ids_src], batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence([ids for ids in ids_tgt], batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

    # Use 0 workers on Windows to avoid multiprocessing issues
    actual_num_workers = 0 if os.name == 'nt' else args.num_workers
    
    offsets = find_offsets(args.data_file, actual_num_workers)
    dataset = LineByLineTextDataset(tokenizer, file_path=args.data_file, offsets=offsets)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_with_tokenizer, num_workers=actual_num_workers
    )

    model.to(args.device)
    model.eval()
    tqdm_iterator = trange(0, desc="Extracting")

    writers = open_writer_list(args.output_file, actual_num_workers) 
    if args.output_prob_file is not None:
        prob_writers = open_writer_list(args.output_prob_file, actual_num_workers)
    if args.output_word_file is not None:
        word_writers = open_writer_list(args.output_word_file, actual_num_workers)

    for batch in dataloader:
        with torch.no_grad():
            worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
            word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, 0, 0, align_layer=args.align_layer, extraction=args.extraction, softmax_threshold=args.softmax_threshold, test=True, output_prob=(args.output_prob_file is not None))
            for worker_id, word_aligns, sent_src, sent_tgt in zip(worker_ids, word_aligns_list, sents_src, sents_tgt):
                output_str = []
                if args.output_prob_file is not None:
                    output_prob_str = []
                if args.output_word_file is not None:
                    output_word_str = []
                for (word_src, word_tgt), prob in word_aligns.items():
                    if word_src != -1 and word_tgt != -1:
                        output_str.append(f'{word_src}-{word_tgt}')
                        if args.output_prob_file is not None:
                            output_prob_str.append(f'{prob:.6f}')
                        if args.output_word_file is not None:
                            output_word_str.append(f'{sent_src[word_src]}<sep>{sent_tgt[word_tgt]}')
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
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--model_type",
        default="auto",
        type=str,
        choices=["auto", "bert", "roberta", "codebert"],
        help="Type of model to use. If 'auto', will try to detect automatically.",
    )
    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.config_name or args.model_name_or_path, cache_dir=args.cache_dir)
    
    # Determine model type if set to auto
    if args.model_type == "auto":
        model_name = args.model_name_or_path.lower()
        if "roberta" in model_name or "codebert" in model_name:
            args.model_type = "roberta"
        else:
            args.model_type = "bert"
    
    # Load the appropriate tokenizer based on model type
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=True
    )
    
    # Create a wrapper model
    model = CodeBERTForWordAlignment(args.model_name_or_path, tokenizer, config)

    # Run word alignment
    word_align(args, model, tokenizer)


if __name__ == "__main__":
    main()