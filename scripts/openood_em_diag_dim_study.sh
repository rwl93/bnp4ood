#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <CUDA_DEVICE_ID>"
    exit 1
}

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    usage
fi

# Check if the argument is an integer
if [[ ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument is not an integer."
    usage
fi

cuda_device_id=$1

echo "Running experiments on GPU $cuda_device_id"

for ((i=1; i<=9; i++))
do
    dim=$((2**i))
    log_path="openood_exps/diag/logs/pca${dim}.log"
    output_filename="openood_exps/diag/results/pca${dim}.result"
    echo "Running experiment with PCA dimension $dim"
    echo "PCA log_path = $log_path"
    echo "PCA output_filename = $output_filename"

    CUDA_VISIBLE_DEVICES=$cuda_device_id python train_em_diag_hdpmm_openood.py \
        --usepca --pca_dim ${dim} \
        --log_path ${log_path} \
        --output_filename ${output_filename}

    log_path="openood_exps/diag/logs/autowhiten${dim}.log"
    output_filename="openood_exps/diag/results/autowhiten${dim}.result"
    echo "Running experiment with autowhiten dimension$dim"
    echo "Autowhiten log_path = $log_path"
    echo "Autowhiten output_filename = $output_filename"

    CUDA_VISIBLE_DEVICES=$cuda_device_id python train_em_diag_hdpmm_openood.py \
        --autowhiten --autowhiten_dim ${dim} \
        --log_path ${log_path} \
        --output_filename ${output_filename}

done
