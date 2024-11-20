def process_file(input_file, output_file1, output_file2):
    # Initialize lists to store the first and second halves
    first_half = []
    second_half = []

    # Open the input file for reading
    with open(input_file, 'r') as infile:
        for line in infile:
            # Remove any leading/trailing whitespace (like newlines)
            line = line.strip()

            # Split the line by the delimiter '|||'
            parts = line.split('|||')

            # Ensure there are exactly two parts, otherwise ignore the line
            if len(parts) == 2:
                first_half.append(parts[0].strip())  # Store first half
                second_half.append(parts[1].strip())  # Store second half

    # Open the output file for writing
    with open(output_file1, 'w') as outfile:
        # Write the first half followed by the second half
        outfile.write("\n".join(first_half))
        outfile.write("\n")  # Add a newline between the two sections
        outfile.write("\n".join(second_half))

    with open(output_file2, 'w') as outfile:
        # Write the first half followed by the second half
        outfile.write("\n".join(first_half))
        outfile.write("\n")  # Add a newline between the two sections
        outfile.write("\n".join(second_half))

# Usage
# input_file = 'en-fr.txt'  # Replace with the path to your input file
# output_file1 = 'Data/EN-FR/input_overlap.in'  # Replace with the path to your output file
# output_file2 = 'Data/EN-FR/label_overlap.out'  # Replace with the path to your output file

input_file = 'tokenized-java-cs.txt'  # Replace with the path to your input file
output_file1 = 'Data/Java-CS/input_overlap.in'  # Replace with the path to your output file
output_file2 = 'Data/Java-CS/label_overlap.out'  # Replace with the path to your output file
process_file(input_file, output_file1, output_file2)
