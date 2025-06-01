
dims='2 8'
Ks='2 4 8'
lambdas='0.1 0.5' 
etas='0.5 0.1'

for lam in $lambdas;do
  for eta in $etas;do
    for d in $dims; do
      for k in $Ks;do
        echo AUC=$auc d=$d K=$k 
        sbatch run_make_dataset.sh $d $k $lam $eta
      done
    done
  done
done	

