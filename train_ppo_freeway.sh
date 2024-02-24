#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=train_agents
#SBATCH --output=alpha_ppo.log   
#SBATCH --partition=dgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


while getopts p:s: flag
do
    case "${flag}" in
        p) policy_name=${OPTARG};;
        s) torch_seed=${OPTARG};;
    esac
done

echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S
echo $CUDA_VISIBLE_DEVICES

srun -n 1 python train_policy.py MinAtar/Freeway-v1 ppo $policy_name $torch_seed