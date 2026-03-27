#!/bin/bash
#SBATCH --job-name=rf_sep_train
#SBATCH --account=cs6953-gpu-np
#SBATCH --partition=dlair-gpu-np
#SBATCH --qos=cs6953-gpu-np
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=train_%j.log

# 1. Source the conda profile to make the 'conda' command work
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Activate your specific environment by name OR path
conda activate FastICA

# 3. Execute code
python -u train_and_validate.py