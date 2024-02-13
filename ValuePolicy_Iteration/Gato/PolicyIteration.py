from GatoMDP import GatoMDP
import numpy as np
from tqdm import tqdm

class PolicyIteration(GatoMDP):
    def __init__(self, gamma=1, theta = 10e-10):
        super().__init__()
        self.gamma = gamma
        self.theta = theta
        self.policy = {}
        self.V = {}
        self.initialize()

    def initialize(self):
        self.generar_estados()
        self.possible_actions()
        self.terminal_states() 
        # Inicializamos una política al azar y los valores de V = 0
        for state in self.states:
            # Elejimos una accion al azar para cada estado
            self.policy[state] = np.random.choice(self.actions[state]) if self.actions[state] else None
            self.V[state] = 0
    
    def policy_evaluation(self):
        while True:
            delta = 0
            for s in tqdm(self.states):
                if s in self.T_states:
                    self.V[s] = self.reward_function(s)
                    continue
                v = 0
                for s_prime in self.possible_next_state(s, self.policy[s]):
                    v += self.reward_function(s_prime) + self.gamma*self.V[s_prime]
                delta = max(delta, abs(v - self.V[s]))
                self.V[s] = v
            if delta < self.theta:
                break

    def policy_improvement(self):
        for s in tqdm(self.states):
            temp = self.policy[s]
            if s in self.T_states:
                continue
            best_action = None
            best_value = float("-inf")
            stable = True
            for a in self.actions[s]:
                expected_value = self.reward_function(s)
                for s_prime in self.possible_next_state(s, a):
                    expected_value += self.transition_function(s) * (self.reward_function(s_prime) + self.gamma * self.V[s_prime])
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = a
            self.policy[s] = best_action
            if temp != self.policy[s]:
                stable =  False
        return stable

    def policy_iteration(self):
        i = 0
        while True:
            print(f"Epoch {i + 1} started...")
            i += 1
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy