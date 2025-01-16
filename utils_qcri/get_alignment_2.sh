#!/bin/bash -l

# Default values for command-line arguments
top_n_translations=5       # Number of top translations (default: 5)
matching_threshold=0.6     # Matching threshold (default: 0.6)
size_threshold=0.4         # Size difference threshold (default: 0.4)
types=1                    # Minimal distinct word size (default: 1)
basePath=""               # Base directory path
layer=""                  # Layer number
dictionary_file_path=""    # Path to dictionary file

# Usage function to display help
usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --inputPath <path>           Base path to input directory. (required)"
    echo "  --layer <int>                Layer number. (required)"
    echo "  --top_n_translations <int>   Number of top translations to consider. (default: 5)"
    echo "  --matching_threshold <float> Matching threshold for semantic alignment. (default: 0.6)"
    echo "  --size_threshold <float>     Size difference threshold for alignment. (default: 0.4)"
    echo "  --types <int>                Minimal distinct word size. (default: 1)"
    echo "  --dictionary <path>          Path to the dictionary file. (required)"
    echo "  -h, --help                   Display this help message and exit."
    echo
    echo "Example:"
    echo "  $0 --inputPath Experiments/model/data/extraction_without_filtering --layer 0 --dictionary Dictionary/en-fr.json"
    exit 0
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --inputPath) basePath="$2"; shift ;;
        --layer) layer="$2"; shift ;;
        --top_n_translations) top_n_translations="$2"; shift ;;
        --matching_threshold) matching_threshold="$2"; shift ;;
        --size_threshold) size_threshold="$2"; shift ;;
        --types) types="$2"; shift ;;
        --dictionary) dictionary_file_path="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$basePath" ]]; then
    echo "Error: --inputPath is required."
    usage
fi

if [[ -z "$layer" ]]; then
    echo "Error: --layer is required."
    usage
fi

if [[ -z "$dictionary_file_path" ]]; then
    echo "Error: --dictionary is required."
    usage
fi

# Construct the full input path with the layer number
clusterDir="${basePath%/*}/layer${layer}/${basePath##*/}/clustering"

# Find the encoder and decoder cluster files in the specified clusterDir
encoder_cluster_file=$(find "$clusterDir" -type f -name "encoder-clusters-kmeans-*.txt" | head -n 1)
decoder_cluster_file=$(find "$clusterDir" -type f -name "decoder-clusters-kmeans-*.txt" | head -n 1)

if [[ -z "$encoder_cluster_file" || -z "$decoder_cluster_file" ]]; then
    echo "Error: Could not find encoder or decoder cluster files in $clusterDir."
    exit 1
fi

# Log detected files
echo "Encoder cluster file: $encoder_cluster_file"
echo "Decoder cluster file: $decoder_cluster_file"

# Create the log file path
log_file="$clusterDir/alignment_output.txt"

# Run the Python alignment script
python -u "code/alignClusters.py" \
    "$encoder_cluster_file" \
    "$decoder_cluster_file" \
    "$dictionary_file_path" \
    "$top_n_translations" \
    "$matching_threshold" \
    "$size_threshold" \
    "$types" | tee "$log_file"

