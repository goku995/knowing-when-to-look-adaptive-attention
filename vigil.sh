#!/bin/bash
#SBATCH -A kanishk
#SBATCH --mem-per-cpu=4096
#SBATCH -n 10
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=kanishk.jain@alumni.iiit.ac.in
#SBATCH --mail-type=END

module load cuda/10.0
module load cudnn/7-cuda-10.0

set -e

python3 train_eval.py
