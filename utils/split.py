import os
import re
import argparse

# Main function to split lines and write to two files
def split_file(input_file, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the full paths for the output files
    output_file1 = os.path.join(output_dir, 'input.in')
    output_file2 = os.path.join(output_dir, 'label.out')

    with open(input_file, 'r') as infile, \
         open(output_file1, 'w') as outfile1, \
         open(output_file2, 'w') as outfile2:
        
        for line in infile:
            # Split the line at '|||'
            parts = line.strip().split('|||')
            if len(parts) == 2:
                # Write the first part to the first output file
                outfile1.write(parts[0].strip() + '\n')
                # Write the second part to the second output file
                outfile2.write(parts[1].strip() + '\n')
            else:
                print(f"Skipping line: {line.strip()} (incorrect format)")
        print(f'Encoder input saved to {outfile1.name}')
        print(f'Decoder input saved to {outfile2.name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a file into two parts based on '|||' delimiter.")
    parser.add_argument('input_file', type=str, help="Path to the input file.")
    parser.add_argument('--output_dir', type=str, help="Directory where the output files will be saved.")

    args = parser.parse_args()
      # Set default output file if not provided
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_file) 

    split_file(args.input_file, args.output_dir)
