#!/bin/bash
# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)
#SBATCH --time=10:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=16   # 8 processor core(s) per GPU
#SBATCH --mem=320G   # maximum memory per node
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="extraction"
#SBATCH --mail-user=xiaoquan@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -A bweng-lab

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# Print host name
hostname
# Check GPU(s)
nvidia-smi
eval "$(conda shell.bash hook)"
# source activate basis_reconstruction_library
conda activate neurox_pip

python3 bash_cluster.py