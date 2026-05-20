#!/bin/bash
# Run this script on the HPC login node BEFORE submitting slurm_cifar10.sh.
# It downloads CIFAR-10 (~170 MB) to /home/fritzsche/cifar10/.
#
# Usage:
#   bash download_cifar10.sh

set -e

DATA_DIR="/home/fritzsche/cifar10"
mkdir -p "$DATA_DIR"

source /home/fritzsche/qkformer/bin/activate

python - <<EOF
import torchvision
torchvision.datasets.CIFAR10(root="$DATA_DIR", train=True,  download=True)
torchvision.datasets.CIFAR10(root="$DATA_DIR", train=False, download=True)
print("CIFAR-10 download complete:", "$DATA_DIR")
EOF
