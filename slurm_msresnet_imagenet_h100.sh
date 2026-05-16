#!/bin/bash
# MS-ResNet-104 ImageNet training — SLURM job with automatic chaining
#
# Each job runs for up to WALL_TIME hours, then resubmits itself until
# TARGET_EPOCHS epochs are finished.
#
# First submission:  sbatch slurm_msresnet_imagenet_h100.sh
# Subsequent jobs chain automatically. Manual resubmission also works —
# it will resume from the latest checkpoint in OUTPUT_DIR.

#SBATCH --job-name=msresnet_imagenet
#SBATCH --partition=normal
#SBATCH --nodelist=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # single task — torchrun spawns 8 processes internally
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0                         # all available memory on the node
#SBATCH --time=24:00:00
#SBATCH --output=/home/fritzsche/MS-ResNet/logs/msresnet_%j.out
#SBATCH --error=/home/fritzsche/MS-ResNet/logs/msresnet_%j.err

# ── User-configurable ─────────────────────────────────────────────────────────
PROJECT_DIR="/home/fritzsche/MS-ResNet"         
DATA_DIR="/nfsscratch/fritzsche/imagenet"
OUTPUT_DIR="/nfsscratch/fritzsche/msresnet_output/imagenet"
LOG_DIR="/home/fritzsche/MS-ResNet/logs"

TARGET_EPOCHS=125      # must match conf/global_settings.py EPOCH
NUM_GPUS=8
WALL_TIME=23           # graceful stop after this many hours (1 h buffer before SLURM kills)

# ── Environment setup ─────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

module purge
module load cuda/12.3
module load gnu12/12.3.0

source /home/fritzsche/qkformer/bin/activate   # TODO: adjust to your venv

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"

# ── Checkpoint auto-detection ─────────────────────────────────────────────────
LATEST_CKPT="$OUTPUT_DIR/checkpoint-latest.pth"
RESUME_ARG=""
COMPLETED_EPOCHS=0

if [[ -f "$LATEST_CKPT" ]]; then
    echo "Found checkpoint: $LATEST_CKPT"
    RESUME_ARG="-resume $LATEST_CKPT"
    COMPLETED_EPOCHS=$(python3 - <<EOF
import torch
try:
    ckpt = torch.load("$LATEST_CKPT", map_location="cpu")
    print(ckpt.get("epoch", 0))
except Exception:
    print(0)
EOF
)
else
    echo "No checkpoint found — starting fresh."
fi

echo "Epochs completed so far: $COMPLETED_EPOCHS / $TARGET_EPOCHS"

if [[ "$COMPLETED_EPOCHS" -ge "$TARGET_EPOCHS" ]]; then
    echo "Training already complete. Exiting."
    exit 0
fi

# ── Training ──────────────────────────────────────────────────────────────────
# Hyperparameters: SGD + momentum, cosine LR, label smoothing, AMP
#   Batch 256 total (32 per GPU × 8), LR 0.1, 125 epochs, time_window=5
cd "$PROJECT_DIR"

timeout $((WALL_TIME * 3600)) \
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    train_amp_hpc.py \
        -net resnet104_fast \
        -b 256 \
        -lr 0.1 \
        -data_path "$DATA_DIR" \
        -output_dir "$OUTPUT_DIR" \
        -workers 10 \
        $RESUME_ARG

TRAIN_EXIT=$?
echo "torchrun exit code: $TRAIN_EXIT"
echo "End time: $(date)"

# ── Auto-resubmit ─────────────────────────────────────────────────────────────
# exit 0   — training finished normally
# exit 124 — WALL_TIME elapsed, checkpoint saved, safe to resubmit
# anything else — genuine error, do not resubmit

if [[ $TRAIN_EXIT -ne 0 && $TRAIN_EXIT -ne 124 ]]; then
    echo "Training failed (exit $TRAIN_EXIT) — not resubmitting."
    exit $TRAIN_EXIT
fi

NEW_COMPLETED=$(python3 - <<EOF
import torch
try:
    ckpt = torch.load("$LATEST_CKPT", map_location="cpu")
    print(ckpt.get("epoch", 0))
except Exception:
    print(0)
EOF
)

echo "Epochs completed after this job: $NEW_COMPLETED / $TARGET_EPOCHS"

if [[ "$NEW_COMPLETED" -lt "$TARGET_EPOCHS" ]]; then
    echo "Submitting next chained job..."
    NEXT_JOB=$(sbatch --parsable "$0")
    echo "  Next job ID: $NEXT_JOB"
else
    echo "Training complete! All $TARGET_EPOCHS epochs finished."
    echo "Best model: $OUTPUT_DIR/best_model.pth"
fi
