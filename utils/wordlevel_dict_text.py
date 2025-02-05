import argparse
import os
import re

def tokenize(code):
    # Regular expression to split by word boundaries and keep special characters
    tokens = re.findall(r'\w+|[^\s\w]', code)
    return tokens

# Function to create the per-word level dictionary from input files
def create_word_level_dictionary(text_file_path, alignment_file_path, output_file_path):
    # Read the text input file
    with open(text_file_path, 'r') as text_file:
        text_input = text_file.read()

    # Read the alignment input file
    with open(alignment_file_path, 'r') as alignment_file:
        alignment_input = alignment_file.read()

    # Prepare to parse the text input
    source_phrases = []
    target_phrases = []

    # Split the text input into source and target phrases
    for line in text_input.strip().split('\n'):
        source, target = line.split('|||')
        source_phrases.append(source.strip())
        target_phrases.append(target.strip())

    # Prepare to parse the alignment input
    alignments = []
    for line in alignment_input.strip().split('\n'):
        alignment = line.split()
        alignments.append([tuple(map(int, pair.split('-'))) for pair in alignment])

    # Create a dictionary to hold counts and probabilities
    word_level_dictionary = {}

    # Initialize the count dictionary to hold total counts
    count_dict = {}

    # Generate the dictionary content based on the alignments
    for phrase_index, alignment in enumerate(alignments):

        # Get the source and target phrases
        source_phrase = source_phrases[phrase_index]
        target_phrase = target_phrases[phrase_index]

        # Split phrases into words using the tokenize function instead of simple split
        source_words = tokenize(source_phrase)
        target_words = tokenize(target_phrase)

        # print(f"Phrase {phrase_index}:")
        # print(f"  Source words: {source_words}")
        # print(f"  Target words: {target_words}")
        # print(f"  Alignment: {alignment}")
    
        # Create mappings for each word in the source to the corresponding word in the target
        for source_index, target_index in alignment:

            if source_index >= len(source_words):
                print(f"Skipping invalid source index {source_index} for phrase {phrase_index}")
                continue
            if target_index >= len(target_words):
                print(f"Skipping invalid target index {target_index} for phrase {phrase_index}")
                continue
            
            source_word = source_words[source_index]
            target_word = target_words[target_index]

            # Count occurrences
            if source_word not in word_level_dictionary:
                word_level_dictionary[source_word] = {}

            if target_word not in word_level_dictionary[source_word]:
                word_level_dictionary[source_word][target_word] = {'Count': 0}

            word_level_dictionary[source_word][target_word]['Count'] += 1

            # Initialize count_dict to hold total words mapped from each source word
            if source_word not in count_dict:
                count_dict[source_word] = 0
            count_dict[source_word] += 1

    # Calculate probabilities and prepare output
    output_lines = []
    for source_word, targets in word_level_dictionary.items():
        for target_word, metrics in targets.items():
            probability = metrics['Count'] / count_dict[source_word]
            output_lines.append(f"{source_word} -> {target_word}: Probability = {probability:.2f}, Count = {metrics['Count']}")

    # Output the word-level dictionary content to a file
    with open(output_file_path, 'w') as dict_file:
        dict_file.write('\n'.join(output_lines) + '\n')

    print(f"{output_file_path} created successfully.")

# Main function to parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a word-level dictionary from text and alignment files.')
    parser.add_argument('text_file', type=str, help='Path to the text input file. A ||| delimited parallel corpus with proper tokenization.')
    parser.add_argument('alignment_file', type=str, help='Path to the alignment input file with pharaoh (i-j) format.')
    parser.add_argument('--output_file', type=str, default=None, help='Path to the output dictionary json file. (Default: <text_file_directory>/dictionary.txt')

    args = parser.parse_args()

    # Set default for output_file if not provided
    if args.output_file is None:
        text_file_dir = os.path.dirname(args.text_file)
        args.output_file = os.path.join(text_file_dir, 'dictionary.txt')

    # Call the function to create the word-level dictionary
    create_word_level_dictionary(args.text_file, args.alignment_file, args.output_file)


