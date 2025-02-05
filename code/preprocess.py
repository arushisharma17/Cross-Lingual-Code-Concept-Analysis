import argparse
import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during {description}:")
        print(result.stderr)
        sys.exit(1)
    print("Success!")
    return result

def main():
    parser = argparse.ArgumentParser(description='Preprocess code corpus for analysis')
    parser.add_argument('--corpus-path', required=True, help='Path to corpus file (e.g., Data/CPP-Cuda/cpp-cuda.txt)')
    parser.add_argument('--lang1', required=True, help='First language (e.g., cpp)')
    parser.add_argument('--lang2', required=True, help='Second language (e.g., cuda)')
    args = parser.parse_args()

    # Extract directory and base filename from corpus path
    corpus_dir = os.path.dirname(args.corpus_path)
    
    # 1. Tokenize corpus
    run_command([
        "python", "-u", "tree-sitter/tokenize_corpus.py",
        args.corpus_path, args.lang1, args.lang2, "--level", "leaf"
    ], "Tokenizing corpus")

    # 2. Run awesome-align
    run_command([
        "awesome-align",
        "--model_name_or_path", "bert-base-multilingual-cased",
        "--data_file", f"{corpus_dir}/tree_sitter_tokenized_leaf.txt",
        "--output_file", f"{corpus_dir}/forward.align",
        "--extraction", "softmax",
        "--batch_size", "32"
    ], "Running awesome-align")

    # 3. Create translation dictionary
    run_command([
        "python", "-u", "utils/wordlevel_dict_text.py",
        args.corpus_path, f"{corpus_dir}/forward.align"
    ], "Creating text translation dictionary")

    run_command([
        "python", "-u", "utils/wordlevel_dict.py",
        args.corpus_path, f"{corpus_dir}/forward.align"
    ], "Creating translation dictionary")

    # 4. Make mapping dictionary
    run_command([
        "python", "-u", "utils/mapping_awesomealign.py",
        args.corpus_path, f"{corpus_dir}/forward.align"
    ], "Creating mapping dictionary")

    # 5. Split corpus
    run_command([
        "python", "-u", "utils/split.py",
        args.corpus_path
    ], "Splitting corpus")

    print("\n=== Preprocessing completed successfully! ===")
    print("You can now proceed with activation extraction.")

if __name__ == "__main__":
    main() 