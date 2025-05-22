"""RMDS for OpenOOD benchmark"""
# Python
import argparse
import logging
import os
import sys
# Third Party
from jaxtyping import Float
import torch
from torch import Tensor
# Modules
from bnp4ood.rmds import rmd_score_fun, compute_rmd_params, compute_indep_rmd_params, indep_rmd_score_fun
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
parser.add_argument("--log_path", type=str, default="mds_rmds.log")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--prefix", type=str, default="")
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
    logger.info("Fit RMDS on ViT Features for OpenOOD")
    logger.info("------------------------------------")
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

    # Setup Initial priors
    logging.info("Calculating RMDS parameters...")
    mu0, L0, muk, J = compute_rmd_params(Nk, sumx, sumxxT)
    indep_mu0, indep_J0, indep_muk, indep_Jk = compute_indep_rmd_params(Nk, sumx, sumxxT)

    # Evaluate
    rmds_output = dict(openood={})
    mds_output = dict(openood={})
    indep_rmds_output = dict(openood={})
    logger.info("Evaluating on OpenOOD benchmark ")
    rmds_aurocs = openood_eval(dataset_dict, (muk, J), rmd_score_fun, model="RMDS", mu0=mu0, L0=L0)
    mds_aurocs = openood_eval(dataset_dict, (muk, J), rmd_score_fun, model="MDS")
    indep_rmds_aurocs = openood_eval(dataset_dict, (indep_muk, indep_Jk), indep_rmd_score_fun,
                                     model="Indep. RMDS", mu0=indep_mu0, J0=indep_J0)
    rmds_output["openood"] = rmds_aurocs
    mds_output["openood"] = mds_aurocs
    indep_rmds_output["openood"] = indep_rmds_aurocs
    logger.info("MDS AUROC:")
    for k, v in mds_aurocs.items():
        logger.info(f"{k}: {v}")
    logger.info("RMDS AUROC:")
    for k, v in rmds_aurocs.items():
        logger.info(f"{k}: {v}")
    logger.info("Independent RMDS AUROC:")
    for k, v in indep_rmds_aurocs.items():
        logger.info(f"{k}: {v}")
    torch.save(rmds_output, os.path.join(args.output_path, args.prefix + "rmds.result"))
    torch.save(mds_output, os.path.join(args.output_path, args.prefix + "mds.result"))
    torch.save(indep_rmds_output, os.path.join(args.output_path, args.prefix + "indep_rmds.result"))


if __name__ == "__main__":
    main(parser.parse_args())
