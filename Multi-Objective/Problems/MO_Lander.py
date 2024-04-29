import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
import torch
from Net import Net

class MO_LANDER:
    def __init__(self):
        self.env = mo_gym.make('mo-lunar-lander-v2')
        self.dim_state = self.env.observation_space.shape[0]
        self.dim_action = self.env.action_space.n
        self.net = Net(self.dim_state, self.dim_action) 
        self.n_var = sum(p.numel() for p in self.net.parameters())
        self.n_obj = 4

    def get_action(self,state):
        action_probs = self.net.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        return action

    
    def evaluate(self, params):
        self.net.set_params(params)
        reward_sums = np.zeros(4)
        for t in range(5):
            state, _ = self.env.reset()
            done = False
            steps = 0
            while not done and steps < 1000:
                action = self.get_action(state)
                state, reward, done, _, _ = self.env.step(action)
                # Sumar la recompensa a la suma correspondiente
                reward_sums += reward
                steps += 1

        # Calcular el promedio de las recompensas
        reward_avgs = reward_sums / 5
        reward_avgs[0] *= -1
        reward_avgs[1] *= -1

        return reward_avgs