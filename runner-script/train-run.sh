#!/bin/bash
# ===============================
# JOB DISCRIPTION HERE: BoloFAST first run
# ===============================

#SBATCH --job-name=evBoloFAST
#SBATCH --output=TrFASTbolo_%j.out
#SBATCH --error=TrFASTbolo_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=18GB

# NOT SURE WHAT THIS DOES
module purge

# Run script
# setting up env variables
export WORKINGDIR=($pwd)/../../..
export DATADIR=$WORKINGDIR/BoloGAN-FCC/input/dataset1


# bind working dir to boloGANtainer and exec train.py or evaluate.py
apptainer exec \
    -B $WORKINGDIR/BoloGAN-FCC:/BoloGAN-FCC \
    --pwd /BoloGAN-FCC/training \
    BoloGANtainer_Plus.sif \
    bash -c "
       python3 train.py \
           -i ../input/dataset1/dataset_1_photons_1.hdf5 \
           -o ../output/dataset1/v1/GANv1_GANv1 \
           -c ../config/config_GANv1.json \
	       --max_iter 100000
    "
