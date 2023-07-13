#!/bin/bash
#SBATCH --account=mayocancerai
##SBATCH --job-name=enc_dec
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mwolff3@uwyo.edu
#SBATCH --time=1-00:00:00
#SBATCH --partition=beartooth
python3 /project/mayocancerai/mwolff3/wolff_code/06232023.py > 06302023.out
