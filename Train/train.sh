#!/bin/bash
#SBATCH -t 3-0
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH -J inp64
#SBATCH --output=inp.out

source activate DL
python -u train.py Celeba Inpaint 48 48 64 64 True True False
