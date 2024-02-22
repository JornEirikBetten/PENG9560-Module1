from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from gymnasium import spaces

class FreewayCNN(BaseFeaturesExtractor):
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
    
class SeaquestCNN(BaseFeaturesExtractor):
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
    
class DoorkeyCNN(BaseFeaturesExtractor):
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