#!/bin/bash

# run the environment creation line only the first time this script runs or run before calling the script:
# if using GPU:
#conda create --name fairness python=3.11 numpy scipy tqdm scikit-learn pytorch pytorch-cuda=11.8 -c pytorch -c nvidia   -y
# if using CPU only:
#conda create --name fairness python=3.11 numpy scipy tqdm scikit-learn pytorch cpuonly -c pytorch  -y


etasettings='equal minority'
dims='2 8'
components='2 4 8'
aucs='0 1 2'
lams='0.1 0.5'

for lam in $lams;do
  for eta in $etasettings;do
    for d in $dims;do 
      for k in $components;do
        for auc in $aucs;do
          echo dim=$d K=$k AUC=$auc lambda=$lam eta=$eta
          sbatch run_synthetic.sh $d $k $auc $lam $eta
        done
      done
    done
  done
done

