#!/bin/sh

# Function to display usage
usage() {
    echo "Usage: $0 <CUDA_DEVICE_ID> [--vitb16]"
    exit 1
}

# Check if exactly one argument is provided
if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    usage
fi

# Check if the argument is an integer
if [[ ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument is not an integer."
    usage
fi
cuda_device_id=$1

if [[ "$#" -eq 1 ]]; then
    feats="vit-b-16"
else
    case $2 in
        --vitb16)
            feats="vit-b-16"
            ;;
        *)
            usage
            ;;
    esac
fi

echo "Running experiments on GPU $cuda_device_id"

if [[ $feats == "vit-b-16" ]]; then
    echo "Running experiments with DeIT ViT-B-16 features"
    feats="vit-b-16"
    featprefix=
else
    echo "Invalid feature type"
    usage
fi

CUDA_VISIBLE_DEVICES=$cuda_device_id python train_em_diag_hdpmm_openood.py \
    --features $feats \
    --log_path openood_exps/diag/logs/${featprefix}vit768.log \
    --output_filename openood_exps/diag/results/${featprefix}vit768.result

CUDA_VISIBLE_DEVICES=$cuda_device_id python train_em_diag_hdpmm_openood.py \
    --autowhiten \
    --features $feats \
    --log_path openood_exps/diag/logs/${featprefix}autowhiten.log \
    --output_filename openood_exps/diag/results/${featprefix}autowhiten.result

CUDA_VISIBLE_DEVICES=$cuda_device_id python train_em_diag_hdpmm_openood.py \
    --usepca --pca_dim 765 \
    --features $feats \
    --log_path openood_exps/diag/logs/${featprefix}pca765.log \
    --output_filename openood_exps/diag/results/${featprefix}pca765.result
