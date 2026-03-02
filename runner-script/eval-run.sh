#!/bin/bash
# ===============================
# JOB DISCRIPTION HERE: BoloFAST evaluation
# ===============================

#SBATCH --job-name=evBoloFAST
#SBATCH --output=EvFASTbolo_%j.out
#SBATCH --error=EvFASTbolo_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --mem=12GB

# NOT SURE WHAT THIS DOES
module purge

# Run script
# setting up env variables
export WORKINGDIR=($pwd)/../../..
export DATADIR=$WORKINGDIR/BoloGAN-FCC/input/dataset1

# bind working dir to boloGANtainer and exec train.py or evaluate.py

#NOTE: number of treads line can be removed
#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
apptainer exec \
    -B $WORKINGDIR/BoloGAN-FCC:/BoloGAN-FCC \
    --pwd /BoloGAN-FCC/training \
    --env OMP_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    BoloGANtainer_Plus.sif \
    bash -c "
        python3 evaluate.py \
            -i ../input/dataset1/dataset_1_photons_1.hdf5 \
            -t ../output/dataset1/v1/GANv1_GANv1 \
	    --checkpoint \
	    --save_h5 \
	    --debug
    "
