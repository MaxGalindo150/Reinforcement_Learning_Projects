import os
import sys
sys.path.insert(0, "Deep_Reinforcement_Learning/DDQN")
from Breakout.DDQN_agent import Wrap, Agent


env_ = 'PongDeterministic-v0'
env = Wrap(env_)
agent = Agent(env, weights_file=None)
agent.train(1000)