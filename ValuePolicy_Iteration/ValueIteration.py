from GatoMDP import GatoMDP
from tqdm import tqdm
import numpy as np

class ValueIteration(GatoMDP):

    def __init__(self, gamma=1, theta = 10e-10):
        super().__init__()
        self.gamma = gamma # Factor de descuento
        self.theta = theta # Criterio de convergencia
        self.policy = {}    
        self.V = {}

    # Inicializamos los valores de V
    def _init_V(self):
        for state in self.states:
            self.V[state] = 0
    

    # Inicializamos la política
    def _init_P(self):
        for state in self.states:
            self.policy[state] = None

    # Inicializamos el MDP
    def initialize(self):
        self.generar_estados()
        self.possible_actions()
        self.terminal_states()
        self._init_V()
        self._init_P()

    # Algoritmo de iteración de valor
    def value_iteration(self):
        self.initialize()
        epoch = 0
        delta = float('inf')
        while True:
            print(f'Epoch: {epoch+1} started...')
            epoch += 1
            delta = 0
            for s in tqdm(self.states):
                if s in self.T_states:
                    self.V[s] = self.reward_function(s)
                    continue
                v = self.V[s]
                new_value = float('-inf')
                for a in self.actions[s]:
                    for s_prime in self.possible_next_state(s, a):
                        #expected_value = self.improved_transition_function(s, a)*(self.reward_function(s_prime) + self.gamma*self.V[s_prime])
                        expected_value = self.transition_function(s)*(self.reward_function(s_prime) + self.gamma*self.V[s_prime])
                self.V[s] = max(new_value, expected_value)
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break

        for s in self.states:
            if s in self.T_states:
                continue
            best_action = None
            best_value = float("-inf")
            for a in self.actions[s]:
                expected_value = self.reward_function(s)
                for s_prime in self.possible_next_state(s, a):
                    #expected_value += self.improved_transition_function(s, a)* (self.reward_function(s_prime) + self.gamma * self.V[s_prime])
                    expected_value += self.transition_function(s) * (self.reward_function(s_prime) + self.gamma * self.V[s_prime])
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = a
            self.policy[s] = best_action

