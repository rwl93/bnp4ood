"""Split features files into a multiple files."""
# Python
import argparse
import os
# Third-party
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--feats_file", type=str, required=True)
parser.add_argument("--num-parts", type=int, default=2, help="Number of parts to split the file into")
parser.add_argument("--outfile_prefix", type=str, default=None, help="Output file prefix")


def main(args):
    if args.outfile_prefix is None:
        args.outfile_prefix = os.path.basename(args.feats_file).split(".")[0] + "-"

    # Detect all file parts
    data = torch.load(args.feats_file, weights_only=False)
    feats = data["feats"]
    labels = data["labels"]
    batch_size = len(feats) // args.num_parts
    for i in range(args.num_parts):
        start = i * batch_size
        end = (i + 1) * batch_size
        if end > len(feats): end = -1
        curr_feats = feats[start:end]
        curr_labels = labels[start:end]
        curr_outfile = f"{args.outfile_prefix}{i}.part"
        print(f"Saving part {i} to {curr_outfile}")
        torch.save(dict(feats=curr_feats, labels=curr_labels), curr_outfile)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)