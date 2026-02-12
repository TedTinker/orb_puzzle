#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
module load singularity
singularity exec communication.sif python orb_puzzle/plotting.py --comp deigo --arg_title ___e_1+e_2+e_3+e_4+e_5+e_6+e_7+e_8+e_9+e_10+e_11+e_12+e_13+e_14+e_15+e_16+e_17+e_18___ --arg_name plotting
