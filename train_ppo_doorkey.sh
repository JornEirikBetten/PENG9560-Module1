#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=train_ppo_doorkey
#SBATCH --output=alpha_doorkey.log   
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

srun -n 1 python train_policy.py MiniGrid-DoorKey-5x5-v0 ppo $policy_name $torch_seed