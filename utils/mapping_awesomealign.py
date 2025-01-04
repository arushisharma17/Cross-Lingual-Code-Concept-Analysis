import argparse
import json
from pathlib import Path
import os
import re

def tokenize(code):
    # Regular expression to split by word boundaries and keep special characters
    tokens = re.findall(r'\w+|[^\s\w]', code)
    return tokens

def parse_sentences_and_alignments(sentence_file, alignment_file):
    mapping_dict = {}

    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    with open(alignment_file, 'r', encoding='utf-8') as f:
        alignments = [line.strip() for line in f.readlines()]

    for i, (sentence, alignment) in enumerate(zip(sentences, alignments)):
        src_sentence, tgt_sentence = sentence.split("|||")
        print(src_sentence)
        src_words = tokenize(src_sentence)
        print(src_words)
        tgt_words = tokenize(tgt_sentence)
        print(tgt_words)

        # Create a dictionary to group alignments by source/target words
        word_alignments = {}
        
        # Parse the alignments
        alignments_list = alignment.split()
        for align in alignments_list:
            src_idx, tgt_idx = map(int, align.split('-'))
            
            # Create alignment key based on source and target words
            if 0 <= src_idx < len(src_words) and 0 <= tgt_idx < len(tgt_words):
                src_word = src_words[src_idx]
                tgt_word = tgt_words[tgt_idx]
                key = (src_word, tgt_word)
                
                if key not in word_alignments:
                    word_alignments[key] = {'src_indices': [], 'tgt_indices': []}
                
                word_alignments[key]['src_indices'].append(src_idx)
                word_alignments[key]['tgt_indices'].append(tgt_idx)

        # Convert word_alignments to the required format
        sentence_alignments = [
            [src_word, tgt_word, align_info['src_indices'], align_info['tgt_indices']]
            for (src_word, tgt_word), align_info in word_alignments.items()
        ]

        if sentence_alignments:
            mapping_dict[i] = sentence_alignments

    return mapping_dict

def main():
    parser = argparse.ArgumentParser(description="Parse sentence alignments and sentences into mapping_dict format.")
    parser.add_argument('sentence_file', type=Path, help="Path to the sentence pairs file.")
    parser.add_argument('alignment_file', type=Path, help="Path to the Pharaoh alignments file.")
    parser.add_argument('--output-file', type=Path, help="Path to save the output mapping_dict as JSON.")

    args = parser.parse_args()

      # Set default for output_file if not provided
    if args.output_file is None:
        text_file_dir = os.path.dirname(args.sentence_file)
        args.output_file = os.path.join(text_file_dir, 'mapping_dict.json')

    # Parse sentences and alignments
    mapping_dict = parse_sentences_and_alignments(args.sentence_file, args.alignment_file)

    # Save mapping_dict to output JSON file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=4)

    print(f"Saved mapping_dict to {args.output_file}")

if __name__ == '__main__':
    main()

