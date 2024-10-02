import re

# Function to add space around punctuation in the code
def add_space_around_punctuation(code):
    # Define a regular expression pattern for punctuation to be spaced
    pattern = r'([();{}])'
    
    # Use regex to replace each occurrence of punctuation with spaces around it
    spaced_code = re.sub(pattern, r' \1 ', code)
    
    # Replace multiple spaces with a single space, just in case
    spaced_code = re.sub(r'\s+', ' ', spaced_code)
    
    # Strip leading/trailing spaces for cleaner formatting
    return spaced_code.strip()

# Function to merge lines from two files, add spaces around punctuation, and combine them into '|||' format
def merge_files_with_spacing(file1, file2, output_file):
    with open(file1, 'r') as infile1, \
         open(file2, 'r') as infile2, \
         open(output_file, 'w') as outfile:
        
        for line1, line2 in zip(infile1, infile2):
            # Add spaces around punctuation in both lines
            line1_spaced = add_space_around_punctuation(line1.strip())
            line2_spaced = add_space_around_punctuation(line2.strip())
            
            # Combine the two lines with '|||'
            merged_line = f"{line1_spaced} ||| {line2_spaced}\n"
            
            # Write the result to the output file
            outfile.write(merged_line)

# Example usage
file1 = 'train.java-cs.txt.cs'  # Replace with your first file path
file2 = 'train.java-cs.txt.java' # Replace with your second file path
output_file = 'java-cs.txt' # Output file for the combined content

merge_files_with_spacing(file1, file2, output_file)
