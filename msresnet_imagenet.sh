#!/bin/bash
#SBATCH --job-name=msresnet_imagenet
#SBATCH --partition=gpu
#SBATCH --nodelist=n-hpc-gz6       # 4× A100 SXM4 40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40         # 10 workers × 4 GPU processes
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/fritzsche6/MS-ResNet/logs/msresnet_%j.out
#SBATCH --error=/work/fritzsche6/MS-ResNet/logs/msresnet_%j.err

# ── Environment setup ────────────────────────────────────────────────────────
module load system/CUDA/11.6.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qkformer        

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH=/work/fritzsche6/qkformer/imagenet   # reuse existing ImageNet copy
OUTPUT_DIR=/work/fritzsche6/MS-ResNet/output
REPO_DIR=/work/fritzsche6/MS-ResNet/

mkdir -p "$OUTPUT_DIR" /work/fritzsche6/msresnet/logs

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"

# ── Resume support ───────────────────────────────────────────────────────────
RESUME_ARG=""
if [ -f "$OUTPUT_DIR/checkpoint-latest.pth" ]; then
    RESUME_ARG="-resume $OUTPUT_DIR/checkpoint-latest.pth"
    echo "Resuming from $OUTPUT_DIR/checkpoint-latest.pth"
else
    echo "No checkpoint found, starting from scratch"
fi

# ── Training ─────────────────────────────────────────────────────────────────
# Hyperparameters match the MS-ResNet-104 setup from Hu et al. 2024:
#   SGD + momentum, cosine LR, label smoothing, AMP — all handled in the script
#   Batch 256 total (64 per GPU), LR 0.1, 125 epochs (conf/global_settings.py)
#   time_window=6 is set globally in models/MS_ResNet.py
cd "$REPO_DIR"

torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_amp_hpc.py \
    -net resnet104 \
    -b 256 \
    -lr 0.1 \
    -data_path "$DATA_PATH" \
    -output_dir "$OUTPUT_DIR" \
    -workers 10 \
    $RESUME_ARG

echo "End time: $(date)"
