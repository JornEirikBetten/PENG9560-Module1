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


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1),
            #nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            observation = observation_space.sample()[None]
            observation = self.permute_state(observation)
            n_flatten = self.cnn(
                th.as_tensor(observation)
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def permute_state(self, state): 
        if state.__class__.__name__ == 'Tensor': 
            state = state.clone().detach()
        else: 
            state = th.tensor(state)
        return state.clone().detach().permute(0, 3, 1, 2).float()
    
    def forward(self, observations) -> th.Tensor:
        observations = self.permute_state(observations)
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

policy_names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

eval_env = gym.make("MinAtar/Freeway-v1")
path = os.getcwd()    
log_path = path + "/training_runs/freeway/ppo/"

total_timesteps = 5_000_000
for i, policy_name in enumerate(policy_names): 
    save_path = log_path + policy_name + "/"
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                             log_path=save_path, eval_freq=100_000,
                             deterministic=True, render=False)
    env = gym.make("MinAtar/Freeway-v1")
    env = Monitor(env, filename=save_path)
    env_checker.check_env(env)
    sb3_common.env_checker.check_env(env)
    in_channels, num_actions = env.observation_space.shape[2], env.action_space.n
    model = PPO("CnnPolicy", env, n_steps=1024, policy_kwargs=policy_kwargs, verbose=1, seed = 1231, torch_seed = 1123+ 123451*(i), device = "auto")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("Training completed.")
    model.save(log_path + policy_name + "_freeway_ppo")

    del model # remove to demonstrate saving and loading

