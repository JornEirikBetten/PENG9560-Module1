#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=train_rl_agenst
#SBATCH --output=training_of_agents.log   
#SBATCH --partition=dgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S
export LD_LIBRARY_PATH=/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}