from typing import Optional
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
        
class RandomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # Extract the optional 'seed' keyword argument (default: None)
        self.seed: Optional[int] = kwargs.pop("seed", None)
        # Initialize the parent class with the remaining arguments
        super().__init__(*args, **kwargs)
        # Create a NumPy random generator using the optional seed
        self.rng = np.random.default_rng(self.seed)

    def _predict(self, observation, deterministic: bool = False):
        """
        Return a random action. If deterministic is True, return the precomputed fixed action.
        Otherwise, sample a new random action from the action space.
        """
        random_action = self.action_space.sample()
        return th.as_tensor(random_action, device=self.device)
    
class CPPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # Extract the optional 'seed' keyword argument (default: None)
        self.seed: Optional[int] = kwargs.pop("seed", None)
        # Initialize the parent class with the remaining arguments
        super().__init__(*args, **kwargs)
        # Create a NumPy random generator using the optional seed
        self.rng = np.random.default_rng(self.seed)

    def _predict(self, observation, deterministic: bool = False):
        """
        Return a random action. If deterministic is True, return the precomputed fixed action.
        Otherwise, sample a new random action from the action space.
        """
        random_action = self.action_space.sample()
        return th.as_tensor(random_action, device=self.device)