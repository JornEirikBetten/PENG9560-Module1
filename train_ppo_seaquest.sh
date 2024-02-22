#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=train_rl_agenst
#SBATCH --output=training_of_agents.log   
#SBATCH --partition=dgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S
echo $CUDA_VISIBLE_DEVICES

srun -n 1 python train_policy.py MinAtar/Seaquest-v1 ppo