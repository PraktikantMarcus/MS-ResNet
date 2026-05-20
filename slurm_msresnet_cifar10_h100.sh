#!/bin/bash
# MS-ResNet-18 CIFAR-10 training — single GPU, H100 cluster
#
# Pre-requisite: download CIFAR-10 to DATA_DIR on the login node before
# submitting.  From the project directory run:
#   python -c "import torchvision; torchvision.datasets.CIFAR10('/nfsscratch/fritzsche/cifar10', download=True)"
#
# Submission:  sbatch slurm_msresnet_cifar10_h100.sh

#SBATCH --job-name=msresnet_cifar10
#SBATCH --partition=normal
#SBATCH --exclude=multigpu            # single GPU — skip the 8-GPU node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/fritzsche/MS-ResNet/logs/msresnet_cifar10_%j.out
#SBATCH --error=/home/fritzsche/MS-ResNet/logs/msresnet_cifar10_%j.err

# ── User-configurable ─────────────────────────────────────────────────────────
PROJECT_DIR="/home/fritzsche/MS-ResNet"
DATA_DIR="/nfsscratch/fritzsche/cifar10"
OUTPUT_DIR="/nfsscratch/fritzsche/msresnet_output/cifar10"
LOG_DIR="/home/fritzsche/MS-ResNet/logs"

# ── Environment setup ─────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

module purge
module load cuda/12.3
module load gnu12/12.3.0

source /home/fritzsche/qkformer/bin/activate

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"

# ── Resume support ────────────────────────────────────────────────────────────
LATEST_CKPT="$OUTPUT_DIR/checkpoint-latest.pth"
RESUME_ARG=""
if [[ -f "$LATEST_CKPT" ]]; then
    echo "Resuming from $LATEST_CKPT"
    RESUME_ARG="-resume $LATEST_CKPT"
else
    echo "No checkpoint found — starting fresh."
fi

# ── Training ──────────────────────────────────────────────────────────────────
# MS-ResNet-18 CIFAR-10 hyperparameters (Hu et al. 2024):
#   SGD + momentum, cosine LR, standard CE, T=5
#   Batch 128, LR 0.1, 200 epochs, weight_decay=5e-4 (set inside the script)
cd "$PROJECT_DIR"

python train_cifar10_hpc.py \
    -net resnet110_cifar10 \
    -b 128 \
    -lr 0.1 \
    -data_path "$DATA_DIR" \
    -output_dir "$OUTPUT_DIR" \
    -workers 4 \
    $RESUME_ARG

echo "End time: $(date)"
