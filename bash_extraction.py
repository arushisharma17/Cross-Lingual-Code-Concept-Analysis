import os
os.environ['HF_HOME'] = '/work/instruction/coms-599-29-f24/group_4_clustering/Cross-Lingual-Code-Concept-Analysis/cache/'

import subprocess

# Define the script path
script_path = "./utils_qcri/activation_extraction_without_filtering.sh"

print(f"Running {script_path}")

# Loop through arguments from 0 to 12
for arg in range(7, 13):  # Range is exclusive, so use 13 to include 12
    print("Starting Experiement...")
    try:
        print(f"Running script with argument: {arg}")
        result = subprocess.run(
            [script_path, str(arg)],  # Convert the integer to a string
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Output for argument {arg}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred for argument {arg}:\n{e.stderr}")
