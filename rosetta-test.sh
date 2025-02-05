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

source /lustre/hdd/LAS/jannesar-lab/owenk/miniconda3/bin/activate neurox_pip

python code/preprocess.py --corpus-path Data/CPP-Cuda/cpp-cuda.txt --lang1 cpp --lang2 cuda

./utils_qcri/activation_extraction_without_filtering_2.sh --model CodeRosetta/CodeRosetta_cpp2cuda_ft  --inputPath Data/CPP-Cuda --layer 0 --sentence_length 512 

./utils_qcri/clustering_2.sh --inputPath Experiments/CodeRosetta_CodeRosetta_cpp2cuda_ft/Data_CPP-Cuda/extraction_without_filtering --layer 0 --clusters 500 --mode visualize

./utils_qcri/get_alignment_2.sh --inputPath Experiments/CodeRosetta_CodeRosetta_cpp2cuda_ft/Data_CPP-Cuda/extraction_without_filtering --layer 0 --dictionary Data/CPP-Cuda/dictionary.json

