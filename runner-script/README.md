# Running BoloGAN-FCC
the scripts `train-run.sh` and `eval-run.sh` run the training and evaluation steps respectively.
IMPORTANT: these scripts are intended for `sbatch` Slurm Workload Manager. 
```bash
sbatch train-run.sh
sbatch eval-run.sh
```

# Container
The training and evaluation requires `BoloGANtainer` apptainer. the apptainer file named `BoloGANtainer_Plus.sif` must in this directory. 

#Regarding input
make sure to change the input dir according to the dataset you have. the one in the script is for the photon dataset1. And should be thought of as a place holder
