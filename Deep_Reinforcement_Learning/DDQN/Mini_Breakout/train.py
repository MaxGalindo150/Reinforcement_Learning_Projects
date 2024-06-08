import gymnasium as gym
# import os
# import sys
# sys.path.insert(0, "Deep_Reinforcement_Learning/DDQN/")
from DDQN_agent import Agent


env = gym.make('MinAtar/Breakout-v0', render_mode='rgb_array')
env.seed(0)
agent = Agent(env, weights_file=None)
agent.train(5000)