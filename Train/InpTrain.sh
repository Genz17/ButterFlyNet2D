#!/bin/bash
#SBATCH -t 3-0
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH -J inp64
#SBATCH --output=inp.out

source activate DL
python -u Inpaint.py 40 50 64 64 6 2 True True Celeba
