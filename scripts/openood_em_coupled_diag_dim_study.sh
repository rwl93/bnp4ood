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
    pca_log_path="openood_exps/coupled_diag/logs/pca${dim}.log"
    pca_output_filename="openood_exps/coupled_diag/results/pca${dim}.result"
    autowhiten_log_path="openood_exps/coupled_diag/logs/autowhiten${dim}.log"
    autowhiten_output_filename="openood_exps/coupled_diag/results/autowhiten${dim}.result"
    echo "Running experiment with dimension $dim"
    echo "PCA log_path = $pca_log_path"
    echo "PCA output_filename = $pca_output_filename"

    CUDA_VISIBLE_DEVICES=$cuda_device_id python train_em_coupled_diag_hdpmm_openood.py \
        --usepca --pca_dim ${dim} \
        --log_path ${pca_log_path} \
        --output_filename ${pca_output_filename}

    echo "Running experiment with dimension $dim"
    echo "Autowhiten log_path = $autowhiten_log_path"
    echo "Autowhiten output_filename = $autowhiten_output_filename"

    CUDA_VISIBLE_DEVICES=$cuda_device_id python train_em_coupled_diag_hdpmm_openood.py \
        --autowhiten --autowhiten_dim ${dim} \
        --log_path ${autowhiten_log_path} \
        --output_filename ${autowhiten_output_filename}
done
