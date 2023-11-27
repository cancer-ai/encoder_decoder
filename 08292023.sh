#!/bin/bash
#SBATCH --account=mayocancerai
##SBATCH --job-name=enc_dec
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mwolff3@uwyo.edu
#SBATCH --time=10:00:00
#SBATCH --partition=teton-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

python3 /project/mayocancerai/mwolff3/wolff_code/08292023.py > 09282023.out

