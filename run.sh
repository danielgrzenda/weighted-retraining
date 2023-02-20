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
module load python/anaconda-2022.05
conda activate weighted-retraining
python -m pip install -e .

model_dir="results/models/unweighted"
mkdir -p "$model_dir"
cd weighted_retraining

# Takes about 20min to train
# Final model is in "logs/train/shapes/lightning_logs/version_0/last.ckpt"`
python train_scripts/train_shapes.py --root_dir="$model_dir" --latent_dim=12 --dataset_path=../data/shapes/squares_G64_S1-20_seed0_R10_mnc32_mxc33.npz --property_key="areas"

