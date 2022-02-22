import numpy as np
import gym
import random 
import time

# Create environment
env = gym.make("FrozenLake-v1")

# Create Q Table

q_table = np.zeros((env.action_space.n, env.observation_space.n))
print(q_table)