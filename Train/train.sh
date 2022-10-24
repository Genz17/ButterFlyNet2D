#!/bin/bash
#SBATCH -t 7-0
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH -J layerTest64Ff
#SBATCH --output=layerTest64Ff.out

source activate DL
# python -u train.py
python -u FTApprox.py layer 6 64 Fourier f True
