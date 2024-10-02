# Function to split lines and write to two separate files
def split_file(input_file, output_file1, output_file2):
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

# Example usage
input_file = 'large_en_de.text'     # Replace with your input file path
output_file1 = 'input.in'  # First half output
output_file2 = 'label.out' # Second half output

split_file(input_file, output_file1, output_file2)
