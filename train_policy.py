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

from stable_baselines3.common.logger import configure
import src as src 

policy_kwargs = dict(
    features_extractor_class=src.FreewayCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

seaquest = "seaquest"
freeway = "freeway"
dqn_algo = "dqn"
ppo_algo = "ppo"

policy_names =["beta", "gamma", "delta", "epsilon"]

eval_env = gym.make("MinAtar/Freeway-v1")
path = os.getcwd()    
log_path = path + "/training_runs/" + freeway + "/" + ppo_algo + "/"

total_timesteps = 5_000_000
for i, policy_name in enumerate(policy_names): 
    save_path = log_path + policy_name + "/"
    eval_callback = EvalCallback(Monitor(eval_env), best_model_save_path=save_path,
                             log_path=save_path, eval_freq=100_000,
                             deterministic=True, render=False)
    env = gym.make("MinAtar/Freeway-v1")
    env = Monitor(env, filename=save_path)
    env_checker.check_env(env)
    sb3_common.env_checker.check_env(env)
    in_channels, num_actions = env.observation_space.shape[2], env.action_space.n
    model = PPO("CnnPolicy", 
                env, 
                n_steps=2048, 
                policy_kwargs=policy_kwargs, 
                ent_coef=1e-3,
                verbose=1, 
                seed = 1231, 
                torch_seed = 1123 + 123451*(i+1), 
                device = "auto")  
    """ model = dqn.DQN("CnnPolicy", 
                    env, 
                    buffer_size=200_000, 
                    policy_kwargs=policy_kwargs, 
                    target_update_interval=50_000, 
                    verbose=1,
                    seed = 1231, 
                    torch_seed = 1123 + 123451*(i+0))   """
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    print("Training completed.")
    model.save(log_path + policy_name + "_" + freeway + "_" + ppo_algo)

    del model # remove to demonstrate saving and loading

