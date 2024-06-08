import gymnasium as gym
from DDQN_agent import DDQN
import torch
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

#3

# Set the seed
seed = 160
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = DDQN(in_channels=4, num_actions=6)
model.load_state_dict(torch.load('Deep_Reinforcement_Learning/DDQN/Mini_Breakout/best_weights_min_seed.pth'))
model.eval()  # Set the model to evaluation mode

env = gym.make('MinAtar/Breakout-v0', render_mode='human')
env.seed(seed)  # Set the seed for the gym environment


#rewards_per_episode = []

#accumulated_rewards = []
state = env.reset()[0]
done = False
truncate = False
env.render()
while not done and not truncate:
    # Convert the state to a tensor and add an extra dimension
    state = np.transpose(state, (2, 0, 1))
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # Get the Q-values from the model
    with torch.no_grad():
        q_values = model(state_tensor)
    # Select the action that maximizes the Q-value
    action = torch.argmax(q_values).item()
    state, reward, done, truncate, _ = env.step(action)
    #accumulated_rewards.append(reward)
   
#rewards_per_episode.append(sum(accumulated_rewards))

# print(f'Puntajes de los 10 experimentos: {rewards_per_episode}')
# print(f'Average reward: {np.mean(rewards_per_episode)}')
# print(f'Varianza {np.var(rewards_per_episode)}')
# print(f'sum {sum(rewards_per_episode)}')