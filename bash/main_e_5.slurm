#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=10
module load singularity
singularity exec communication.sif python orb_puzzle/main.py --comp deigo --arg_name e_5 --target_entropy -1 --alpha_normal 0.5 --eta 0  --agents $agents_per_job --previous_agents $previous_agents
