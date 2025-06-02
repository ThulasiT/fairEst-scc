# fairEst-scc
A semi-supervised framework for accurate fairness metric estimation under selection bias using sub-class-conditional invariance (SCC-invariance). This repo contains code for the paper "A Semi-Supervised Approach to Improving Fairness Estimates
Under Sample Selection Bias" (paper link to come).

# run the environment creation line only the first time this script runs or run before calling the script:
# if using GPU:
conda create --name fairness python=3.11 numpy scipy tqdm scikit-learn pytorch pytorch-cuda=11.8 -c pytorch -c nvidia   -y
# if using CPU only:
conda create --name fairness python=3.11 numpy scipy tqdm scikit-learn pytorch cpuonly -c pytorch  -y

Activate the environment:
source activate fairness

For synthetic data experiments:
Use run_make_all_datasets.sh to create datasets for the required set of parameters
Once the datasets are made, use run_all_synthetic.sh to run our proposed methods on the created synthetic data

demo.ipynb can be used to estimate fairness metrics for a single set of parameters

The methods listed in the paper correspond to the following variables:
    GIL - Corrected (estimated) //
    GIL* - Corrected (oracle)   //
    GNIL1 - corrected_l (estimated) //
    GNIL2 - corrected_l (estimateNG) //
    GNIL* - corrected_l (oracle) //

Use geneticVariant_experiments.py to run the experiment in the case study
