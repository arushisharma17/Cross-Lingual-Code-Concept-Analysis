import argparse
import json

from pathlib import Path

from collections import Counter

def get_pieces(line):
    pieces = []
    end_idx = len(line)
    for _ in range(3):
        sep_idx = line[:end_idx].rfind("|||")
        pieces.append(line[sep_idx+3:end_idx])
        end_idx = sep_idx
    pieces.append(line[:end_idx])
    return list(reversed(pieces))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dataset', type=Path, required=True)
    parser.add_argument('--src-sentences', type=Path, required=True)
    parser.add_argument('--tgt-dataset', type=Path, required=True)
    parser.add_argument('--tgt-sentences', type=Path, required=True)

    parser.add_argument('--mapping-dict', type=Path, required=True)

    parser.add_argument('--minimum-frequency', type=int, default=5)
    parser.add_argument('--maximum-frequency', type=int, default=50)
    parser.add_argument('--delete-frequency', type=int, default=500000)

    parser.add_argument('--output-src-file-prefix', type=str, required=True)
    parser.add_argument('--output-tgt-file-prefix', type=str, required=True)

    args = parser.parse_args()

    min_freq = args.minimum_frequency
    max_freq = args.maximum_frequency
    del_freq = args.delete_frequency

    print ("Min: ", min_freq, " Max: ", max_freq, " Del: ", del_freq)

    mapping = {}
    with open(args.mapping_dict) as f:
        print("Loading mapping")
        mapping = json.load(f)

    pair_counts = Counter()
    for sentence_idx in mapping:
        for alignment in mapping[sentence_idx]:
            # Each alignment is [src_word, tgt_word, src_indices, tgt_indices]
            src_word, tgt_word, src_idxs, tgt_idxs = alignment
            
            # We'll count each aligned pair once per occurrence
            pair_counts[(src_word, tgt_word)] += 1

    print("Number of unique pairs: {}".format(len(pair_counts)))

    with open(args.src_sentences) as fp:
        src_sentences = json.load(fp)

    with open(args.tgt_sentences) as fp:
        tgt_sentences = json.load(fp)

    with open(args.output_src_file_prefix+"_min_"+str(min_freq)+"_max_"+str(max_freq)+'-sentences.json', 'w') as fp:
        json.dump(src_sentences, fp, ensure_ascii=False)

    with open(args.output_tgt_file_prefix+"_min_"+str(min_freq)+"_max_"+str(max_freq)+'-sentences.json', 'w') as fp:
        json.dump(tgt_sentences, fp, ensure_ascii=False)

    filtered_output_src_dataset = []
    filtered_output_tgt_dataset = []
    src_output_word_counts = Counter()
    tgt_output_word_counts = Counter()


    currPairCount = Counter()
    maxskip = 0
    maxskips = set()
    minskip = 0
    minskips = set()
    delskip = 0
    delskips = set()


    src_word_counts = Counter()
    tgt_word_counts = Counter()
    with open(args.src_dataset) as src_f, open(args.tgt_dataset) as tgt_f:
        print ("Loading src and tgt datasets")
        src_dataset = json.load(src_f)
        tgt_dataset = json.load(tgt_f)

        token_to_src_dataset = {}
        for dataset_idx, (key, activations) in enumerate(src_dataset):
            token, _, sentence_idx, token_idx = key.split("|||")
            src_word_counts[token] += 1
            sentence_idx = int(sentence_idx)
            token_idx = int(token_idx)
            token_to_src_dataset[(sentence_idx, token_idx)] = dataset_idx

        token_to_tgt_dataset = {}
        for dataset_idx, (key, activations) in enumerate(tgt_dataset):
            token, _, sentence_idx, token_idx = key.split("|||")
            tgt_word_counts[token] += 1
            sentence_idx = int(sentence_idx)
            token_idx = int(token_idx)
            
            token_to_tgt_dataset[(sentence_idx, token_idx)] = dataset_idx

        print ("Constructing output datasets")
        for raw_sentence_idx in mapping:
            sentence_idx = int(raw_sentence_idx)
            for alignment in mapping[raw_sentence_idx]:
                src_word, tgt_word, src_idxs, tgt_idxs = alignment
                
                # For each aligned pair of indices
                for src_idx, tgt_idx in zip(src_idxs, tgt_idxs):
                    src_key = (sentence_idx, src_idx)
                    tgt_key = (sentence_idx, tgt_idx)
                    
                    if src_key not in token_to_src_dataset or tgt_key not in token_to_tgt_dataset:
                        continue

                    # Apply frequency filtering
                    if pair_counts[(src_word, tgt_word)] > del_freq:
                        delskip += 1
                        delskips.add((src_word, tgt_word))
                    elif pair_counts[(src_word, tgt_word)] < min_freq:
                        minskip += 1
                        minskips.add((src_word, tgt_word))
                    else:
                        currPairCount[(src_word, tgt_word)] += 1
                        
                        if currPairCount[(src_word, tgt_word)] <= max_freq:
                            # Add to filtered output
                            key, activations = src_dataset[token_to_src_dataset[src_key]]
                            filtered_output_src_dataset.append([key, activations])
                            
                            key, activations = tgt_dataset[token_to_tgt_dataset[tgt_key]]
                            filtered_output_tgt_dataset.append([key, activations])
                        else:
                            maxskip += 1
                            maxskips.add((src_word, tgt_word))

    print("Writing datasets...")
    with open(args.output_src_file_prefix+"_min_"+str(min_freq)+"_max_"+str(max_freq)+"_del_"+str(del_freq)+"-dataset.json", 'w') as fp:
        json.dump(filtered_output_src_dataset, fp, ensure_ascii=False)

    with open(args.output_tgt_file_prefix+"_min_"+str(min_freq)+"_max_"+str(max_freq)+"_del_"+str(del_freq)+"-dataset.json", 'w') as fp:
        json.dump(filtered_output_tgt_dataset, fp, ensure_ascii=False)


    print ("Limit Max types: ", maxskips)
    print ("Skipped Min types: ", minskips)
    print ("Skipped frequent types: ", delskips)

    print ("PAIRWISE STATISTICS:")
    print ("Pairs skipped based on Max freq: ", maxskip)
    print ("Pairs skipped based on Min freq: ", minskip)
    print ("Pairs skipped based on Del freq: ", delskip)
    print ("Unique Pairs skipped based on Max freq: ", len(maxskips))
    print ("Unique Pairs skipped based on Min freq: ", len(minskips))
    print ("Unique Pairs skipped based on Del freq: ", len(delskips))

    print ("SOURCE SIDE STATISTICS:")
    print ("Total word tokens before dropping: ", sum(src_word_counts.values()))
    print ("Total word types before dropping: ", len(src_word_counts))

    print ("Remaining Tokens: ", sum(src_output_word_counts.values()))
    print ("Remaining Types: ", len(src_output_word_counts))

    print ("\nTARGET SIDE STATISTICS:")
    print ("Total word tokens before dropping: ", sum(tgt_word_counts.values()))
    print ("Total word types before dropping: ", len(tgt_word_counts))
    print ("Remaining Tokens: ", sum(tgt_output_word_counts.values()))
    print ("Remaining Types: ", len(tgt_output_word_counts))

if __name__ == '__main__':
    main()
