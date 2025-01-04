import os
os.environ['HF_HOME'] = "./cache"

import subprocess

# Define the base arguments and iterate over the second parameter
base_args = ["1000", "visualize"]
script_path = "./utils_qcri/clustering.sh"

# Loop through values for the second parameter (0 to 12)
for i in range(8, 13):  # Inclusive range 0-12
    # Construct the arguments in the meaningful order
    args = [base_args[0], str(i), base_args[1]]
    try:
        print(f"Running: {script_path} {' '.join(args)}")
        result = subprocess.run(
            [script_path, *args],  # Pass ordered arguments
            check=True,            # Raise an exception if the command fails
            text=True,             # Capture output as a string
            capture_output=True    # Capture both stdout and stderr
        )
        # Print outputs
        print(f"Output for iteration {i}:")
        print(result.stdout)
        if result.stderr:
            print(f"Warnings or errors for iteration {i}:")
            print(result.stderr)
        print(f"Completed iteration {i}.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error during iteration {i}:")
        print(e)
        print("Script output:")
        print(e.stdout)
        print("Script error:")
        print(e.stderr)
