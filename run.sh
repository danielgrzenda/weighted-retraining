#!/bin/sh

#SBATCH --job-name=weighted-retraining
#SBATCH --account=pi-chard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# TO USE V100 specify --constraint=v100
# TO USE RTX600 specify --constraint=rtx6000
#SBATCH --constraint=v100   # constraint job runs on V100 GPU use
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu
#SBATCH --cpus-per-task=1   # set this to the desired number of threads

# LOAD MODULES
#module load tensorflow
#module load cuda/10.2
#module load cudnn

# TODO only need to do this if we aren't on cs or dsi cluster
#module load python/anaconda-2022.05

conda activate weighted-retraining
python -m pip install -e .

model_dir="results/models/uniform"
mkdir -p "$model_dir"

gpu="--gpu"  # change to "" if no GPU is to be used
seed=0

python weighted_retraining/train_scripts/train_shapes.py \
    --root_dir="$model_dir/shapes" \
    --seed="$seed" \
    $gpu \
    --latent_dim=2 \
    --dataset_path=data/shapes/squares_G64_S1-20_seed0_R10_mnc32_mxc33.npz \
    --property_key=areas \
    --max_epochs=20 \
    --weight_type=uniform \
    --rank_weight_k=0.01 \
    --beta_final=10.0 --beta_start=1e-6 \
    --beta_warmup=1000 --beta_step=1.1 --beta_step_freq=10 \
    --batch_size=16
