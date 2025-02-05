import argparse
import json
from pathlib import Path
import os
import re

def parse_sentences_and_alignments(sentence_file, alignment_file):
    mapping_dict = {}

    # Read sentence pairs from file
    with open(sentence_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    # Read alignments from file
    with open(alignment_file, 'r', encoding='utf-8') as f:
        alignments = [line.strip() for line in f.readlines()]

    # Loop through sentences and alignments and parse them
    for i, (sentence, alignment) in enumerate(zip(sentences, alignments)):
        src_sentence, tgt_sentence = sentence.split("|||")
        src_words = tokenize(src_sentence)
        print(src_words)
        tgt_words = tokenize(tgt_sentence)
        print(tgt_words)

        alignments_list = alignment.split()
        sentence_alignments = []

        # Parse alignment pairs and map words
        for align in alignments_list:
            try:
                src_idx, tgt_idx = map(int, align.split('-'))

                # Validate indices
                if src_idx < 0 or src_idx >= len(src_words):
                    print(f"Warning: Adjusting invalid source index {src_idx} for sentence {i}")
                    src_word = "<UNK>"
                else:
                    src_word = src_words[src_idx]

                if tgt_idx < 0 or tgt_idx >= len(tgt_words):
                    print(f"Warning: Adjusting invalid target index {tgt_idx} for sentence {i}")
                    tgt_word = "<UNK>"
                else:
                    tgt_word = tgt_words[tgt_idx]

                sentence_alignments.append([src_word, tgt_word, [src_idx], [tgt_idx]])
            except ValueError:
                print(f"Skipping invalid alignment format '{align}' in sentence {i}")
                continue

        mapping_dict[i] = sentence_alignments

    return mapping_dict

def tokenize(code):
    # Regular expression to split by word boundaries and keep special characters
    tokens = re.findall(r'\w+|[^\s\w]', code)
    return tokens

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

