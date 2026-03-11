#!/usr/bin/env bash

# Run from BoloGAN-FCC/condor; must keep using relative path to run on different machines
cd $(dirname ${BASH_SOURCE[0]})/../training

# ENV SETUP; NOTE: To be ran on INFN machines; Also, LCG_105a_cuda is a temporary choice.. TODO: Update
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh
# installing quickstats b.c. it's missing from LCG_105a_cuda
pip install quickstats --break-system-packages

echo $@

task=$1
input=$2
output=$3
loading=$4

model=`echo $output | cut -d '_' -f 1`
config_mask=`echo $output | cut -d '_' -f 2-1000`
config_mask=`echo $config_mask | cut -d '.' -f 1`
config=`echo $config_mask | cut -d '-' -f 1`
mask=`echo $config_mask | cut -d '-' -f 2 | cut -d 'M' -f 2`
prep=`echo $config_mask | cut -d '-' -f 3 | cut -d 'P' -f 2`
label_scheme=`echo $config_mask | cut -d '-' -f 4 | cut -d 'L' -f 2`
split_energy=`echo $config_mask | cut -d '-' -f 5 | cut -d 'S' -f 2`

echo input=$input
echo output=$output
echo mask=$mask
echo prep=$prep
echo loading=$loading
echo label_scheme=$label_scheme
echo split_energy=$split_energy

if [[ $mask == ?(n)+([0-9]) ]]; then
    version='v2'
    addition="--mask=${mask//n/-}"
#if [[ $mask == ?(n)+([0-9]) ]]; then
#    version='v3'
#    train_addition="--mask=${mask//n/-} --add_noise"
else
    version='v1'
    train_addition=""
fi

if [[ ! -z "$prep" ]]; then
    train_addition="$train_addition -p $prep"
    evaluate_addition="$evaluate_addition -p $prep"
fi

if [[ ! -z "$loading" ]]; then
    train_addition="$train_addition $loading"
    evaluate_addition="$evaluate_addition $loading"
fi

if [[ ! -z "$label_scheme" ]]; then
    train_addition="$train_addition --label_scheme $label_scheme"
fi

if [[ ! -z "$split_energy" ]]; then
    train_addition="$train_addition --split_energy_position $split_energy"
    evaluate_addition="$evaluate_addition --split_energy_position $split_energy"
fi

ds=`echo $input | grep -oP '(?<=input/dataset).'`
if [[ "$ds" = "2" ]]; then
    evaluate_addition="$evaluate_addition --normalise"
fi

if [[ ${task} == *'train'* ]]; then
    #command="python train.py -i ${input} -m ${model} -o ../output/dataset${ds}/${version}/${output} -c ../config/config_${config}.json ${train_addition}"
    command="python3 train.py -i ${input} -m ${model} -o ../output/dataset${ds}/${version}/${output} -c ../config/config_${config}.json ${train_addition} --max_iter 1000000"
else
    command="python3 evaluate.py -i ${input} -t ../output/dataset${ds}/${version}/${output} --checkpoint --debug --save_h5 ${evaluate_addition}"
fi
echo $command
eval $command
cd -
unset mask prep config config_mask model train_addition evaluate_addition loading label_scheme ds
