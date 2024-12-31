#!/bin/bash -l

clusters=$1 # specify the number of clusters
layer=$2
mode=$3
activation_folder=$4

if [ "$mode" == "visualize" ]; then
    echo "Preparing for visualize"
    outputDir="visualize_${clusters}/layer${layer}" # specify the path to the output file 
else
    echo "Preparing for alignment"
    outputDir="layer${layer}" # specify the path to the output file 
fi

# # Delete the output directory if it exists
# if [ -d "$outputDir" ]; then
#     rm -rf "$outputDir"
# fi

mkdir -p $outputDir 

conda activate neurox_pip
vocab_file="${activation_folder}/activationsOutput_no_filtering_${layer}/encoder-processed-vocab-${layer}.npy" # specify the path to the vocab file from the activation extraction step
point_file="${activation_folder}/activationsOutput_no_filtering_${layer}/encoder-processed-point-${layer}.npy" # specify the path to the point file from the activation extraction step
prefix="encoder"

python -u code/create_kmeans_clustering.py -v $vocab_file -p $point_file -o $outputDir -pf $prefix -k $clusters


conda activate neurox_pip
vocab_file="${activation_folder}/activationsOutput_no_filtering_${layer}/decoder-processed-vocab-${layer}.npy" # specify the path to the vocab file from the activation extraction step
point_file="${activation_folder}/activationsOutput_no_filtering_${layer}/decoder-processed-point-${layer}.npy" # specify the path to the point file from the activation extraction step
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

