import argparse
import json
from pathlib import Path

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
        src_words = src_sentence.strip().split()
        tgt_words = tgt_sentence.strip().split()

        alignments_list = alignment.split()
        sentence_alignments = []

        # Parse alignment pairs and map words
        for align in alignments_list:
            src_idx, tgt_idx = map(int, align.split('-'))
            src_word = src_words[src_idx]
            tgt_word = tgt_words[tgt_idx]
            sentence_alignments.append([src_word, tgt_word, [src_idx], [tgt_idx]])

        mapping_dict[i] = sentence_alignments

    return mapping_dict

def main():
    parser = argparse.ArgumentParser(description="Parse sentence alignments and sentences into mapping_dict format.")
    parser.add_argument('--sentence-file', type=Path, required=True, help="Path to the sentence pairs file.")
    parser.add_argument('--alignment-file', type=Path, required=True, help="Path to the Pharaoh alignments file.")
    parser.add_argument('--output-file', type=Path, required=True, help="Path to save the output mapping_dict as JSON.")

    args = parser.parse_args()

    # Parse sentences and alignments
    mapping_dict = parse_sentences_and_alignments(args.sentence_file, args.alignment_file)

    # Save mapping_dict to output JSON file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=4)

    print(f"Saved mapping_dict to {args.output_file}")

if __name__ == '__main__':
    main()

