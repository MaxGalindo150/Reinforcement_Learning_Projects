import numpy as np
import json

class GatoMDP:
    def __init__(self):
        self.states = set()
        self.T_states = set() # Terminal states
        self.actions = {}
        #self.policy = open("Policies/policyIterationGato.json", "r")
        #self.policy = json.load(self.policy)

    def generar_estados(self):
        """
        Se generan los estados posibles e imposibles del juego
        """
        # 0: vacio, 
        # 1: X, 
        # 2: O
        def _brute_states():

            all_board_conf = set()
            for value in np.ndindex((3,3,3,3,3,3,3,3,3)):
                state = tuple(value)
                all_board_conf.add(state)
            return all_board_conf
    
        # Funcion para encontrar dos ganaodres al mismo tiempo
        def _check_2_win(state):
            """
            Esta función encuentra si hay dos ganadores al mismo tiempo
            """
            count_h, count_v = 0, 0
            for i in range(3):
                if (state[i*3] == state[i*3+1] == state[i*3+2] and state[i*3] == 1) or (state[i*3] == state[i*3+1] == state[i*3+2] and state[i*3] == 2):
                    count_h += 1
                if (state[i] == state[3+i] == state[6+i] and state[i] == 1) or (state[i] == state[3+i] == state[6+i] and state[i] == 2):
                    count_v += 1
                if count_h == 2 or count_v == 2:
                    return True
                return False

        # Inicializamos con todos los estados posibles
        self.states = _brute_states()
        
        # Eliminamos los estados que no son posibles
        for state in self.states.copy():
            # Si existen más X que O, se elimina el estado
            if state.count(1) > state.count(2):
                self.states.remove(state)
            # La diferencia de X y O no puede ser mayor a 1
            elif abs(state.count(1) - state.count(2)) > 1:
                self.states.remove(state)
            # Existen dos ganadores al mismo tiempo
            elif _check_2_win(state):
                self.states.remove(state)
    
    def terminal_states(self):
        """
        Se toman todos los estados posibles y se actualizan los estados terminales
        """

        for state in self.states:
            if self.win(state):
                self.T_states.add(state)
            elif state.count(0) == 0:
                self.T_states.add(state)

    def possible_actions(self):
        """
        Toma todos los posibles estados y obtiene las posibles acciones en cada estado
        """

        for state in self.states:
            self.actions[state] = None
            if state not in self.T_states:
                self.actions[state] = []
                for i in range(9):
                    if state[i] == 0:
                        self.actions[state].append(i) # Coincide con el indice de la casilla desocupada

    def transition_function(self, state):
        """"
        Esta función toma un estado y regresa la probabilidad de cada posible siguiente estado
        """
        # Si el estado es terminal retornamos 0
        if state in self.T_states:
            return 0
        else:
            return 1/(len(self.actions[state])-1) #Todos los siguientes estados son equiprobables
            
    def reward_function(self, state):
        """
        Esta función toma un estado y regresa la recompensa de ese estado
        """

        if self.win(state) == 1:
            return 1
        elif self.win(state) == 2:
            return -1
            
        return 0
    
    def win(self, state):
        for i in range(3):
            if state[i*3] == state[i*3+1] == state[i*3+2] and state[i*3] != 0:
                return state[i*3]
            if state[i] == state[3+i] == state[6+i] and state[i] != 0:
                return state[i]
        if state[4] == state[6] == state[2] and state[4] != 0:
            return state[4]
        elif state[0] == state[4] == state[8] and state[0] != 0:
            return state[0]
        else:
            return False
    
    def possible_next_state(self, state, action):
        """
        Esta función toma un estado y una acción y regresa el siguiente estado
        """
        new_state = list(state)
        new_state[action] = 1
        if self.win(new_state):
            return []
        possible_next_states = []
        for i, case in enumerate(new_state):
            next_new_state = new_state.copy()
            if case == 0:
                next_new_state[i] = 2
                possible_next_states.append(tuple(next_new_state))
        
        return possible_next_states
    
    def improved_transition_function(self, state, action):
        """
        Esta función toma un estado y una acción y regresa la probabilidad de cada posible siguiente estado
        """
        # Si el estado es terminal retornamos 0
        if state in self.T_states:
            return 0
        else:
            return 1 if action == self.policy[state] else 0 # Es decir, la politica manda

        
                