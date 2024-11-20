#!/bin/bash -l

conda activate neurox_pip

# Print usage instructions
usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --inputPath <path>         Path to input sentence files. (default: Experiments/google-t5_t5-base/Data_EN-DE/layer3/extraction_with_filtering)"
  echo "  --layer <int>              Layer to cluster activations for (default: 0)."
  echo "  --clusters <int>           Number of clusters to create, (K parameter) (default: 500)."
  echo "  --mode <string>            Goal for the clustering. Impacts final output format (default: visualize)"
  echo "  -h, --help                 Display this help message and exit."
  echo
  echo "Example:"
  echo "  bash $0 --inputPath Data/Java-CS --model bert-base-uncased --layer 2"
  exit 0
}



while [[ $# -gt 0 ]]; do
  case $1 in
    --inputPath)
      inputPath="$2"
      shift
      shift
      ;;
    --layer)
      layer="$2"
      shift
      shift
      ;;
    --clusters)
      clusters="$2"
      shift
      shift
      ;;
    --mode)
      mode="$2"
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

inputPath="${inputPath:-Experiments/google-t5_t5-base/Data_EN-DE/layer3/extraction_with_filtering}"
clusters="${clusters:-500}"
mode="${mode:-visualize}"
layerName=$(basename $(dirname "$inputPath"))

# Output the extracted directory name
if [ "$mode" == "visualize" ]; then
    echo "Preparing for visualize"
    outputDir="$inputPath/clustering" # specify the path to the output file 
else
    echo "Preparing for alignment"
    outputDir="$inputPath/clustering" # specify the path to the output file 
fi

# Delete the output directory if it exists
if [ -d "$outputDir" ]; then
    rm -rf "$outputDir"
fi

mkdir -p $outputDir
echo "outputDir: $outputDir"
echo "clusters: $clusters"

conda activate neurox_pip
vocab_file="$inputPath/encoder-processed-vocab.npy" # specify the path to the vocab file from the activation extraction step
point_file="$inputPath/encoder-processed-point.npy" # specify the path to the point file from the activation extraction step
prefix="encoder"

python -u code/create_kmeans_clustering.py -v $vocab_file -p $point_file -o $outputDir -pf $prefix -k $clusters


conda activate neurox_pip
vocab_file="$inputPath/decoder-processed-vocab.npy" # specify the path to the vocab file from the activation extraction step
point_file="$inputPath/decoder-processed-point.npy" # specify the path to the point file from the activation extraction step
prefix="decoder"

python -u code/create_kmeans_clustering.py -v $vocab_file -p $point_file -o $outputDir -pf $prefix -k $clusters


if [ "$mode" == "visualize" ]; then
    # Example logic to combine files
    # Assuming the output files are "<prefix>-clusters-kmeans-<clusters>.txt"
    encoder_file="$outputDir/encoder-clusters-kmeans-$clusters.txt"
    decoder_file="$outputDir/decoder-clusters-kmeans-$clusters.txt"
    combined_file="$outputDir/clusters-$clusters.txt"

    # Check if both files exist before combining
    if [ -f "$encoder_file" ] && [ -f "$decoder_file" ]; then
        # Combine the two files into one, with a header
        {
            cat "$encoder_file"
            cat "$decoder_file"
        } > "$combined_file"

        echo "Combined output written to: $combined_file"
    else
        echo "One or both files do not exist. Cannot combine."
    fi
fi