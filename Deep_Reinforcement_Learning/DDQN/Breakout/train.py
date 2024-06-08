# import os
# import sys
# sys.path.insert(0, "Deep_Reinforcement_Learning/DDQN/")
from DDQN_agent import Wrap, Agent


env_ = 'ALE/Breakout-v5'
env = Wrap(env_)
agent = Agent(env, weights_file=None)
agent.train(5000)