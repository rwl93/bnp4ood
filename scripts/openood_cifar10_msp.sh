#!/bin/sh

# Function to display usage
usage() {
    echo "Usage: $0 <CUDA_DEVICE_ID> [--resnet18]"
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
    feats="ResNet18_32x32"
fi

for ((i=2; i<= $#; i++)); do
    case ${!i} in
        --resnet18)
            feats="ResNet18_32x32"
            ;;
        *)
            usage
            ;;
    esac
done

echo "Running experiments on GPU $cuda_device_id"

if [[ $feats == "ResNet18_32x32" ]]; then
    echo "Running experiments with ResNet18 32x32 features"
    featprefix=
else
    echo "Invalid feature type"
    usage
fi


for iter in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$cuda_device_id python train_msp_openood.py \
        --features $feats \
        --id-data cifar10 \
        --data-root cifar10_feats/ \
        --model-iter $iter \
        --log_path cifar10_openood_exps/msp/logs/${featprefix}s${iter}.log \
        --output_path cifar10_openood_exps/msp/results \
        --prefix ${featprefix}s${iter}_

    CUDA_VISIBLE_DEVICES=$cuda_device_id python train_msp_openood.py \
        --autowhiten \
        --features $feats \
        --id-data cifar10 \
        --data-root cifar10_feats/ \
        --model-iter $iter \
        --log_path cifar10_openood_exps/msp/logs/${featprefix}s${iter}_autowhiten.log \
        --output_path cifar10_openood_exps/msp/results \
        --prefix ${featprefix}s${iter}_autowhiten_
done
