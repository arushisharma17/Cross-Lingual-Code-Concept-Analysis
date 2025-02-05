#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=18   # 18 processor core(s) per node 
#SBATCH --mem=64G   # maximum memory per node
#SBATCH --job-name="preprocessing-test"
#SBATCH --mail-user=owenk@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"   # Separate file for errors

source /work/LAS/jannesar-lab/owenk/Cross-Lingual-Code-Concept-Analysis/miniconda3/bin/activate neurox_pip

python code/preprocess.py --corpus-path Data/CPP-Cuda/cpp-cuda.txt --lang1 cpp --lang2 cuda

./utils_qcri/activation_extraction_without_filtering_2.sh --model CodeRosetta/CodeRosetta_cpp2cuda_ft  --inputPath Data/CPP-Cuda --layer 0 --sentence_length 512 

./utils_qcri/clustering_2.sh --inputPath Experiments/CodeRosetta_CodeRosetta_cpp2cuda_ft/Data_CPP-Cuda/extraction_without_filtering --layer 0 --clusters 500 --mode visualize

./utils_qcri/get_alignment_2.sh --inputPath Experiments/CodeRosetta_CodeRosetta_cpp2cuda_ft/Data_CPP-Cuda/extraction_without_filtering --layer 0 --dictionary Data/CPP-Cuda/dictionary.json