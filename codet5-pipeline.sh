#!/bin/bash

#SBATCH --time=24:00:00   
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=18   
#SBATCH --mem=64G   
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova    
#SBATCH --job-name="preprocessing-test"
#SBATCH --mail-user=owenk@iastate.edu   
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

# Create logs directory
mkdir -p logs

model_name="Salesforce/codet5-base"
layer=12
short_model_name="${model_name#*/}"


# Run preprocessing
python code/preprocess.py --corpus-path Data/CPP-Cuda/cpp-cuda.txt --lang1 cpp --lang2 cuda

# Run activation extraction
./utils_qcri/activation_extraction_without_filtering_2.sh \
    --model $model_name \
    --inputPath Data/CPP-Cuda \
    --layer $layer \
    --sentence_length 512

# Run clustering
./utils_qcri/clustering_2.sh \
    --inputPath Experiments/$short_model_name/Data_CPP-Cuda/extraction_without_filtering \
    --layer $layer \
    --clusters 350 \
    --mode visualize

# Extract embeddings and compute centroids
python code/extract_embeddings.py \
    --model_name $model_name \
    --corpus_path Data/CPP-Cuda/cpp-cuda.txt \
    --cluster_file1 Experiments/$short_model_name/Data_CPP-Cuda/extraction_without_filtering/layer0/encoder/clusters.txt \
    --cluster_file2 Experiments/$short_model_name/Data_CPP-Cuda/extraction_without_filtering/layer0/decoder/clusters.txt \
    --output_dir Data/CPP-Cuda \
    --layer $layer

# Run alignment with centroids
python code/alignClusters.py \
    Experiments/$short_model_name/Data_CPP-Cuda/extraction_without_filtering/layer0/encoder/clusters.txt \
    Experiments/$short_model_name/Data_CPP-Cuda/extraction_without_filtering/layer0/decoder/clusters.txt \
    Data/CPP-Cuda/dictionary.json \
    5 0.3 0.5 10 \
    --centroid_similarities Data/CPP-Cuda/centroid_similarities.json