from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import ppo, DQN
import minigrid
import gymnasium as gym 
import minatar 
import os 
import numpy as np 
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import ImgObsWrapper


path = os.getcwd() 
dqn_freeway_path = path + "/training_runs/freeway/dqn/"
ppo_freeway_path = path + "/training_runs/freeway/ppo/"
dqn_seaquest_path = path + "/training_runs/seaquest/dqn/"
ppo_doorkey_path = path + "/training_runs/doorkey/ppo/"
policy_names = ["alpha", "beta", "gamma", "delta", "epsilon"]


for i, policy_name in enumerate(policy_names): 
    freeway = Monitor(gym.make("MinAtar/Freeway-v1"))
    ppo_freeway = ppo.PPO.load(ppo_freeway_path + policy_name + "/best_model.zip", freeway, device="auto")
    mean_returns, std_returns = evaluate_policy(ppo_freeway, freeway, n_eval_episodes=100)
    print("PPO Freeway " + policy_name + f" mean return: {mean_returns} +- {1.96*(std_returns/np.sqrt(100))}")
    """seaquest = Monitor(gym.make("MinAtar/Seaquest-v1"))
    dqn_seaquest = DQN.load(dqn_seaquest_path + policy_name + "/best_model.zip", seaquest, device="auto")
    mean_returns, std_returns = evaluate_policy(dqn_seaquest, seaquest, n_eval_episodes=100)
    print("DQN Seaquest " + policy_name + f" mean return: {mean_returns} +- {1.96*(std_returns/np.sqrt(100))}")
    
    freeway = Monitor(gym.make("MinAtar/Freeway-v1"))
    dqn_freeway = DQN.load(dqn_freeway_path + policy_name + "/best_model.zip", freeway, device="auto")
    mean_returns, std_returns = evaluate_policy(dqn_freeway, freeway, n_eval_episodes=100)
    print("DQN Freeway " + policy_name + f" mean return: {mean_returns} +- {1.96*(std_returns/np.sqrt(100))}")
    
    doorkey = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode = "rgb_array")
    doorkey = Monitor(ImgObsWrapper(doorkey))
    ppo_doorkey = ppo.PPO.load(ppo_doorkey_path + policy_name + "/best_model.zip", doorkey, device = "auto")
    mean_returns, std_returns = evaluate_policy(ppo_doorkey, doorkey, n_eval_episodes=100)
    print("PPO Door Key " + policy_name + f" mean return: {mean_returns} +- {1.96*(std_returns/np.sqrt(100))}")"""