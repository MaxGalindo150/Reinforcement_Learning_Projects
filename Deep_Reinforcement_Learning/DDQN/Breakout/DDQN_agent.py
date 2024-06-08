import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class Net(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, num_actions)
        self.state_value1 = nn.Linear(3136, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
     
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))
        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)
        output = state_value + (action_value - action_value.mean())
        return output


class Wrap(gym.Wrapper):
    def __init__(self, env_, render_mode= 'rgb_array', repeat=4, training=True):
        env = gym.make(env_, render_mode=render_mode)
        
        super(Wrap, self).__init__(env)

        self.image_shape = (84, 84)
        self.repeat = repeat
        self.lives = env.ale.lives()
        self.frame_buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = training

    def step(self, action):
        total_reward = 0
        done = False 

        for i in range(self.repeat):
            if not self.training:
                time.sleep(0.01)
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            current_lives = info['lives']

            if self.training:
                if current_lives < self.lives:
                    total_reward += -1
                    self.lives = current_lives


            self.frame_buffer.append(obs)
            
            if done:
                break
        
        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_obs(max_frame)
        #max_frame = max_frame.to(self.device)
        
        #total_reward = total_reward.to(self.device)
        
        #done = done.to(self.device)
        

        return max_frame, total_reward, done, truncated, info
    
    def reset(self):
        self.frame_buffer = []
        obs = self.env.reset()[0]
        self.lives = self.env.ale.lives()
        obs = self.process_obs(obs)

        return obs


    def process_obs(self, obs):
        
        img = Image.fromarray(obs)
        img = img.resize(self.image_shape)
        img = img.convert('L')
        img = np.array(img)
        img = img / 255.0

        #img = img.to(self.device)


        return img
        


class Agent:
    def __init__(self, env, weights_file=None):
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.best_reward = -np.inf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env

        
        self.n_actions = env.action_space.n
        self.n_channels = 1


        self.policy_net = Net(self.n_channels, self.n_actions).to(self.device)
        self.target_net = Net(self.n_channels, self.n_actions).to(self.device)

        if weights_file is not None:
            self.policy_net.load_state_dict(torch.load(weights_file))
            self.target_net.load_state_dict(torch.load(weights_file))

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)



        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # Use policy_net to select the best action for the next state
            #non_final_next_states = non_final_next_states.permute(0, 3, 1, 2)
            next_state_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
           

            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def plot_durations(self):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Result')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.savefig('train.png')

    def train(self, num_episodes):
        for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get its state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            episode_reward = 0
            for t in count():
                #state = state.permute(0, 3, 1, 2)
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                episode_reward += reward.item()
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    torch.save(self.policy_net.state_dict(), 'weights.pth')
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)


                if done:
                    self.episode_durations.append(episode_reward)
                    break 

        print('Complete')
        self.plot_durations()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    print(device)
    env_ = 'ALE/Breakout-v5'
    env = Wrap(env_, training=True, render_mode='rgb_array', device=device)
    agent = Agent(env, device, weights_file=None)
    agent.train(100)