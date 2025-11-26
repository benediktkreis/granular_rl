from setuptools import setup, find_packages

setup(name='sand_gym',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gymnasium', 'mujoco', 'stable-baselines3', 'sb3-contrib', 'tqdm', 'tensorboard', 'robosuite', 'wandb', 'huggingface-hub']
)
 
