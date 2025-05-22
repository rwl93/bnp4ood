"""Expectation Maximization Hierarchical DPMM for OpenOOD"""
# Python
import argparse
import logging
import sys
# Third Party
import torch
# Modules
import bnp4ood.tied_dpmm as tied_dpmm
from bnp4ood.openood_dataset_utils import (
    setup_dataloaders, get_sufficient_stats, openood_eval
    NUM_CLASSES, CIFAR10_NUM_CLASSES, CIFAR100_NUM_CLASSES,
    DATASET_FEATFILES, CIFAR10_FEATFILES, CIFAR100_FEATFILES,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
# Logging
parser.add_argument("--log_path", type=str, default="em_dpmm.log")
parser.add_argument("--output_filename", type=str, default="em_dpmm.result")
# Model
parser.add_argument("--autowhiten", action="store_true")
parser.add_argument("--autowhiten_factor", type=float, default=1e-7)
parser.add_argument("--autowhiten_dim", type=int, default=0)
parser.add_argument("--usepca", action="store_true")
parser.add_argument("--pca_dim", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--features", type=str, default="vit-b-16")
# Dataset
parser.add_argument("--id-data", type=str, default="imagenet")
parser.add_argument("--model-iter", type=int, default=-1)
parser.add_argument("--data-root", type=str, default="./")


def main(args):
    # Logging
    fh = logging.FileHandler(args.log_path)
    logger.addHandler(fh)
    logger.info("Fitting Tied covariance DPMM on ViT Features for OpenOOD")
    logger.info("--------------------------------------------------------")
    logger.info("Arguments:")
    logger.info(args)

    # Load datasets
    if args.id_data == "imagenet":
        dset_featfiles = DATASET_FEATFILES
        num_classes = NUM_CLASSES
    elif args.id_data == "cifar10":
        dset_featfiles = CIFAR10_FEATFILES
        num_classes = CIFAR10_NUM_CLASSES
    elif args.id_data == "cifar100":
        dset_featfiles = CIFAR100_FEATFILES
        num_classes = CIFAR100_NUM_CLASSES
    else:
        raise ValueError(f"Unknown dataset {args.id_data}")

    dataset_dict, id_feats, id_labels = setup_dataloaders(
        model_iter=args.model_iter,
        dataset_featfiles=dset_featfiles,
        data_root=args.data_root,
        K=num_classes,
        batch_size=args.batch_size,
        autowhiten=args.autowhiten,
        autowhiten_factor=args.autowhiten_factor,
        autowhiten_dim=args.autowhiten_dim,
        use_pca=args.usepca,
        pca_dim=args.pca_dim,
        features=args.features,
    )
    init_feats, _ = next(iter(dataset_dict["id"]["train"]))
    D = init_feats.shape[-1]

    Nk, sumx, sumxxT, _ = get_sufficient_stats(
        X=id_feats, Y=id_labels, K=num_classes,
    )

    logging.info("Calculating tied cov DPMM params...")
    prior_params, posterior_params = tied_dpmm.compute_tied_params(Nk, sumx, sumxxT)

    # Move to device
    prior_params = tuple([p.to(DEVICE) for p in prior_params])
    posterior_params = tuple([p.to(DEVICE) for p in posterior_params])

    # Evaluate
    logger.info("Evaluating on OpenOOD benchmark")
    params = (prior_params, posterior_params)
    def compute_scores(X, prior_params, posterior_params, return_preds=False):
        return tied_dpmm.compute_scores(
            prior_params, posterior_params, X,
            return_preds=return_preds,
            Ns=Nk,
        )
    aurocs = openood_eval(dataset_dict, params, compute_scores, model="DPMM")
    output = dict(openood = aurocs)
    logger.info("Tied DPMM AUROC:")
    for k, v in aurocs.items():
        logger.info(f"{k}: {v}")
    torch.save(output, args.output_filename)


if __name__ == "__main__":
    main(parser.parse_args())
