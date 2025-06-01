#!/bin/bash

#SBATCH --job-name=fairness
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --mem=128G
#SBATCH --cpus-per-task 8
#SBATCH --time=24:00:00
#SBATCH --output=../logs/fair_%j.out
#SBATCH --error=../logs/fair_%j.err


# activate envorment 
source activate fairness

d=$1
K=$2
auc=$3
lam=$4
eta=$5
output='synthetic_results_test20250531'
data='../synth_datasets'

srun python synthetic_experiments.py --estimateNG --estimate --etaType $eta --lam $lam --comp $K --dim $d --auc $auc --output $output --dataset_path $data

