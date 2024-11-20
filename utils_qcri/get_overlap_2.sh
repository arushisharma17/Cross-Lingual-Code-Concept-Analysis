#!/bin/bash -l

# Print usage instructions
usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --clusters_path <path>       Path to the encoder clusters file."
  echo "  --output_path <path>         Path to the output file. (default: overlap.txt)"
  echo "  --clusters_threshold <float> Threshold for multilingual or overlapping clusters. (default: 0.4)"
  echo "  --sentences_threshold <int>  Threshold for sentence splits. (default: 232)"
  echo "  --unique_tokens <int>        Minimal number of tokens a cluster should have. (default: 3)"
  echo "  -h, --help                   Display this help message and exit."
  echo
  echo "Example:"
  echo "  bash $0 --clusters_path --output_path overlap_new.txt --clusters_threshold 0.5"
  exit 0
}

# Default values
output_path="overlap.txt"
clusters_threshold="0.4"
sentences_threshold="232"
unique_tokens="3"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --clusters_path)
      clusters_path="$2"
      shift
      shift
      ;;
    --output_path)
      output_path="$2"
      shift
      shift
      ;;
    --clusters_threshold)
      clusters_threshold="$2"
      shift
      shift
      ;;
    --sentences_threshold)
      sentences_threshold="$2"
      shift
      shift
      ;;
    --unique_tokens)
      unique_tokens="$2"
      shift
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Run the Python script with the specified arguments
python -u "code/get_overlapping_clusters.py" \
  --cluster_file "$clusters_path" \
  --output_path "$output_path" \
  --clusters_threshold "$clusters_threshold" \
  --sentences_threshold "$sentences_threshold" \
  --unique_tokens "$unique_tokens"
