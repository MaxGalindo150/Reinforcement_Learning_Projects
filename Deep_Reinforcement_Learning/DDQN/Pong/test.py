import gymnasium as gym
import os
import sys
sys.path.insert(0, "Deep_Reinforcement_Learning/DDQN")
from Breakout.DDQN_agent import Wrap, Net
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import time


# Set the environment
env_ = 'PongDeterministic-v0'
env = Wrap(env_, training=False, render_mode='human')
num_actions = env.action_space.n
model = Net(in_channels=1, num_actions=num_actions)
model.load_state_dict(torch.load('Deep_Reinforcement_Learning/DDQN/Pong/weights_pong.pth'))
model.eval()  
done = False
truncate = False


accumulated_rewards = []
state = env.reset()
for t in count():
    # Convert the state to a tensor and add an extra dimension
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Get the Q-values from the model
    with torch.no_grad():
        q_values = model(state_tensor)
    # Select the action that maximizes the Q-value
    action = torch.argmax(q_values).item()
    observation, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncate
    
    if terminated:
        state = None
    else:
        state = observation
    
    accumulated_rewards.append(reward)
    if done:
        break




