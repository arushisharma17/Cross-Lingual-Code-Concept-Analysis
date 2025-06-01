import argparse
import os
from tree_sitter import Parser, Language
import tqdm
import subprocess
import sys

def build_languages(left_lang, right_lang):
    """Build the tree-sitter language parsers."""
    try:
        # Check if the language repositories exist and have the required files
        for lang in [left_lang, right_lang]:
            lang_dir = f'tree-sitter/tree-sitter-{lang}'
            if not os.path.exists(lang_dir):
                raise FileNotFoundError(f"Language directory not found: {lang_dir}")
            if not os.path.exists(os.path.join(lang_dir, 'src')):
                raise FileNotFoundError(f"Source directory not found in {lang_dir}")

        # Create build directory if it doesn't exist
        os.makedirs('build', exist_ok=True)

        # Build the languages
        Language.build_library(
            'build/my-languages.so',
            [
                f'tree-sitter/tree-sitter-{left_lang}',
                f'tree-sitter/tree-sitter-{right_lang}'
            ]
        )
        
        return True
    except Exception as e:
        print(f"Error building languages: {str(e)}")
        return False

def extract_tokens(source_code: bytes, parser: Parser, level='leaf'):
    tree = parser.parse(source_code)
    root_node = tree.root_node

    def recurse(node):
        tokens = []
        if level == 'line' and node.type.endswith('statement'):
            # For statements, consider the entire line as a token
            tokens.append(node.text.decode('utf-8').strip())
        elif len(node.children) == 0:
            tokens.append(node.text.decode('utf-8'))
        elif level == 'leaf':
            for child in node.children:
                tokens.extend(recurse(child))
        return tokens

    return recurse(root_node)

def process_parallel_corpus(corpus_file, output_file, left_parser, right_parser, level='leaf'):
    with open(corpus_file, 'r') as f, open(output_file, 'w') as output:
        for line in tqdm.tqdm(f):
            try:
                # Split the parallel functions
                left_func, right_func = line.strip().split(' ||| ')
    
                # Parse and tokenize the left function
                left_tokens = extract_tokens(left_func.encode('utf-8'), left_parser, level=level)
    
                # Parse and tokenize the C# function
                right_tokens = extract_tokens(right_func.encode('utf-8'), right_parser, level=level)
    
                # Join tokens with space and write to output
                output.write(f"{' '.join(left_tokens)} ||| {' '.join(right_tokens)}\n")
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                print(f"Problematic line: {line}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Tokenize a ||| delimited parallel corpus of functions in two different languages.')
    parser.add_argument('corpus_file', type=str, help='Path to the input parallel corpus file.')
    parser.add_argument('left_lang', type=str, help='The name of the language that appears on the left of the ||| delimiter.')
    parser.add_argument('right_lang', type=str, help='The name of the language that appears on the right of the ||| delimiter.')
    parser.add_argument('--level', type=str, choices=['leaf', 'line'], default='leaf',
                        help='Tokenization level: "leaf" for individual tokens, "line" for higher-level constructs.')
    parser.add_argument('--output_file', type=str, help='Path to the output file for formatted tokens.')
    
    args = parser.parse_args()

    # Set default output file if not provided
    if not args.output_file:
        input_dir = os.path.dirname(args.corpus_file)
        output_file_name = f"tree_sitter_tokenized_{args.level}.txt"
        args.output_file = os.path.join(input_dir, output_file_name)

    # Build the language parsers
    print("Building language parsers...")
    if not build_languages(args.left_lang, args.right_lang):
        sys.exit(1)

    try:
        # Load the languages
        LEFT_LANGUAGE = Language('build/my-languages.so', args.left_lang)
        RIGHT_LANGUAGE = Language('build/my-languages.so', args.right_lang)

        # Initialize parsers
        left_parser = Parser()
        right_parser = Parser()
        
        left_parser.set_language(LEFT_LANGUAGE)
        right_parser.set_language(RIGHT_LANGUAGE)

        print('Processing corpus...')
        process_parallel_corpus(args.corpus_file, args.output_file, left_parser, right_parser, level=args.level)
        print('Done')
        print(f'Tokenized corpus written to {args.output_file}')

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
