#!/bin/sh
#SBATCH --job-name=synth
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --mem=64G
#SBATCH --cpus-per-task 8
#SBATCH --time=1:00:00
#SBATCH --output=logs/fair_%j.out
#SBATCH --error=logs/fair_%j.err

# source activate fairness

# srun python make_datasets.py --dim $1 --k $2 --lam $3 --eta $4 
python make_datasets.py --dim $1 --k $2 --lam $3 --eta $4 

