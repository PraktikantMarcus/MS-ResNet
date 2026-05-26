#!/bin/bash
# MS-ResNet-110 CIFAR-100 training — single GPU, H100 cluster
#
# Submission:  sbatch slurm_msresnet_cifar100_h100.sh

#SBATCH --job-name=msresnet_cifar100
#SBATCH --partition=normal
#SBATCH --exclude=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/fritzsche/MS-ResNet/logs/msresnet_cifar100_%j.out
#SBATCH --error=/home/fritzsche/MS-ResNet/logs/msresnet_cifar100_%j.err

# ── User-configurable ─────────────────────────────────────────────────────────
PROJECT_DIR="/home/fritzsche/MS-ResNet"
DATA_DIR="/nfsscratch/fritzsche/cifar100"
OUTPUT_DIR="/nfsscratch/fritzsche/msresnet_output/cifar100"
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
    # Extract completed epochs from checkpoint to decide whether to resubmit
    DONE=$(python -c "import torch; c=torch.load('$LATEST_CKPT',map_location='cpu'); print(c['epoch'])" 2>/dev/null || echo "0")
    echo "Resuming from $LATEST_CKPT (epoch $DONE completed)"
    RESUME_ARG="--resume $LATEST_CKPT"
else
    DONE=0
    echo "No checkpoint found — starting fresh."
fi

# ── Training ──────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

python train_hpc.py \
    --dataset cifar100 \
    --data_path "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers 4 \
    $RESUME_ARG

TRAIN_EXIT=$?

# ── Job chaining: resubmit if training is not complete ────────────────────────
if [[ $TRAIN_EXIT -eq 0 ]]; then
    DONE=$(python -c "import torch; c=torch.load('$OUTPUT_DIR/checkpoint-latest.pth',map_location='cpu'); print(c['epoch'])" 2>/dev/null || echo "0")
    if [[ $DONE -lt 200 ]]; then
        echo "Epoch $DONE / 200 complete — resubmitting job."
        sbatch "$0"
    else
        echo "Training complete at epoch $DONE."
    fi
fi

echo "End time: $(date)"
