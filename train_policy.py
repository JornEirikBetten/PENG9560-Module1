import os 
import torch.nn.functional as f 
import stable_baselines3.dqn as dqn  
import stable_baselines3.common as sb3_common
from stable_baselines3.common import env_checker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
import gymnasium as gym 
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import sys 
from stable_baselines3.common.logger import configure
import src as src 
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)

environment = sys.argv[1]
algorithm = sys.argv[2]
policy_name = sys.argv[3]
torch_seed = sys.argv[4]

print("Training agents in " + environment + " using the " + algorithm + " algorithm.")
# Set environment and learning algorithm
#environment = "MinAtar/Freeway-v1" # "MinAtar/Seaquest-v1"
#algorithm = "dqn"
policy_kwargs = dict(
    features_extractor_class=src.FreewayCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# Setting policy 
if environment == "MinAtar/Freeway-v1": 
    policy_kwargs = dict(
        features_extractor_class=src.FreewayCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
if environment == "MinAtar/Seaquest-v1": 
    policy_kwargs = dict(
        features_extractor_class=src.SeaquestCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

if environment == "MiniGrid-DoorKey-5x5-v0": 
    policy_kwargs = dict(
        features_extractor_class=src.DoorkeyCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

if environment == "MiniGrid-DoorKey-6x6-v0": 
    policy_kwargs = dict(
        features_extractor_class=src.DoorkeyCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

if environment == "MiniGrid-DoorKey-16x16-v0": 
    policy_kwargs = dict(
        features_extractor_class=src.DoorkeyCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    

seaquest = "seaquest"
freeway = "freeway"
dqn_algo = "dqn"
ppo_algo = "ppo"

if environment == "MinAtar/Freeway-v1": 
    env_name = freeway 
if environment == "MinAtar/Seaquest-v1": 
    env_name = seaquest 
if environment == "MiniGrid-DoorKey-5x5-v0": 
    env_name = "doorkey"
if environment == "MiniGrid-DoorKey-6x6-v0": 
    env_name = "doorkey6"
if environment == "MiniGrid-DoorKey-16x16-v0": 
    env_name = "doorkey16"
if algorithm == "dqn": 
    algo = dqn_algo
if algorithm == "ppo": 
    algo = ppo_algo





if environment == "MiniGrid-DoorKey-16x16-v0": 
    eval_env = gym.make(environment, render_mode="rgb_array")
    eval_env = ImgObsWrapper(eval_env)
elif environment == "MiniGrid-DoorKey-6x6-v0": 
    eval_env = gym.make(environment, render_mode="rgb_array")
    eval_env = ImgObsWrapper(eval_env)
elif environment == "MiniGrid-DoorKey-5x5-v0": 
    eval_env = gym.make(environment, render_mode="rgb_array")
    eval_env = ImgObsWrapper(eval_env)
else: 
    eval_env = gym.make(environment)

path = os.getcwd()    
log_path = path + "/training_runs/" + env_name + "/" + algo + "/"

total_timesteps = 10_000_000

save_path = log_path + policy_name + "/"
eval_callback = EvalCallback(Monitor(eval_env), best_model_save_path=save_path,
                            log_path=save_path, eval_freq=100_000,
                            deterministic=True, render=False)

if environment == "MiniGrid-DoorKey-16x16-v0": 
    env = gym.make(environment, render_mode = "rgb_array")
    env = ImgObsWrapper(env)
elif environment == "MiniGrid-DoorKey-6x6-v0": 
    env = gym.make(environment, render_mode="rgb_array")
    env = ImgObsWrapper(env)
elif environment == "MiniGrid-DoorKey-5x5-v0": 
    env = gym.make(environment, render_mode="rgb_array")
    env = ImgObsWrapper(env)
else: 
    env = gym.make(environment)
        
env = Monitor(env, filename=save_path)
""" env_checker.check_env(env)
sb3_common.env_checker.check_env(env)
in_channels, num_actions = env.observation_space.shape[2], env.action_space.n """
if algo == "ppo": 
    model = PPO("CnnPolicy", 
                env, 
                n_steps=2048, 
                policy_kwargs=policy_kwargs, 
                ent_coef=1e-4,
                verbose=1, 
                seed = 1231, 
                torch_seed = torch_seed, 
                device = "auto")   
if algo == "dqn": 
    if environment == "MiniGrid-DoorKey-5x5-v0": 
        model = dqn.DQN("CnnPolicy", 
                        env, 
                        buffer_size=50_000, 
                        policy_kwargs=policy_kwargs, 
                        target_update_interval=10_000, 
                        learning_rate=1e-2, 
                        learning_starts=50_000, 
                        exploration_fraction=0.50, 
                        verbose=1, 
                        seed = 1231, 
                        torch_seed=torch_seed)
    else: 
        model = dqn.DQN("CnnPolicy", 
                    env, 
                    buffer_size=200_000, 
                    policy_kwargs=policy_kwargs, 
                    target_update_interval=50_000, 
                    verbose=1,
                    seed = 1231, 
                    torch_seed = torch_seed) 
model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
print("Training completed.")
model.save(log_path + policy_name + "_" + env_name + "_" + algo)


