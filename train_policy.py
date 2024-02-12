import gymnasium as gym
import torch 
import stable_baselines3.dqn as dqn  

def permute_state(state): 
        return (torch.tensor(state, device=self.device).permute(2, 0, 1)).unsqueeze(0).float()

env = gym.make("MinAtar/Freeway-v1")
n_obs_space, n_action_space = env.observation_space, env.action_space
#action_value_estimator = dqn.CnnPolicy(n_obs_space, n_action_space, 0.00025, normalize_images=False)

model = dqn.DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)
model.save("freeway")

del model # remove to demonstrate saving and loading

model = dqn.DQN.load("dqn_cartpole")

obs, info = env.reset()
episode = 0 
max_episodes = 100 

while (episode < max_episodes):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        episode += 1 
print(model.logger())