import argparse
import os
from tree_sitter import Language, Parser
import tqdm

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
                print(f'{line=}')
                break

def main():
    parser = argparse.ArgumentParser(description='Tokenize a ||| delimited parallel corpus of functions in two different languages.')
    parser.add_argument('corpus_file', type=str, help='Path to the input parallel corpus file.')
    parser.add_argument('output_file', type=str, nargs='?', 
                        help='Path to the output file for formatted tokens. Defaults to tree_sitter_tokenized_{level}.txt in the input file\'s directory.',
                        default=None)
    parser.add_argument('left_lang', type=str, help='The name of the language that appears on the left of the ||| delimiter in your parallel corpus.', default='java')
    parser.add_argument('right_lang', type=str, help='The name of the language that appears on the right of the ||| delimiter in your parallel corpus.', default='c_sharp')
    parser.add_argument('--level', type=str, choices=['leaf', 'line'], default='leaf',
                        help='Tokenization level: "leaf" for individual tokens, "line" for higher-level constructs (e.g., statements).')

    args = parser.parse_args()

    # Set default output file if not provided
    if args.output_file is None:
        input_dir = os.path.dirname(args.corpus_file)
        output_file_name = f"tree_sitter_tokenized_{args.level}.txt"
        args.output_file = os.path.join(input_dir, output_file_name)

    # Load the compiled languages
    Language.build_library(
        'build/my-languages.so',  # Shared library to store parsers
        [
            f'tree-sitter/tree-sitter-{args.left_lang}',
            f'tree-sitter/tree-sitter-{args.right_lang}'
        ]
    )

    # .replace() needed for Weird C# issue. This could come back to bite me in the butt.
    LEFT_LANGUAGE = Language('build/my-languages.so', f'{args.left_lang}')
    RIGHT_LANGUAGE = Language('build/my-languages.so', f'{args.right_lang}') 
    
    left_parser = Parser()
    left_parser.set_language(LEFT_LANGUAGE)
    
    right_parser = Parser()
    right_parser.set_language(RIGHT_LANGUAGE)

    print('Processing corpus...')
    process_parallel_corpus(args.corpus_file, args.output_file, left_parser, right_parser, level=args.level)
    print('Done')
    print(f'Tokenized corpus written to {args.output_file}')

if __name__ == '__main__':
    main()
