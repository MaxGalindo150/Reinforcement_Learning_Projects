import json
import random

class Gato:
    def __init__(self, agent = "ValueIteration"):
        self.agent = agent
        self.board = (0,0,0,0,0,0,0,0,0)
        self.agent_symbol = "O"
        self.computer_symbol = "X"
        self.policy = {}
        if agent == "ValueIteration":
            path = "ValuePolicy_Iteration/Policies/ValueIteration_POLICY_TTT.json"
        elif agent == "PolicyIteration":
            path = "ValuePolicy_Iteration/Policies/PolicyIteration_POLICY_TTT.json"
        with open(path, 'r') as json_file:
            self.policy = json.load(json_file)

    def computer_move(self):
        valid_moves = [i for i in range(9) if self.board[i] == 0]
        if valid_moves:
            move = random.choice(valid_moves)
            self.make_move(move, 1)  # Place a 1

    def agent_move(self):
        state = self.board
        move = self.policy[str(state)]
        self.make_move(move, 2)  # Place a 2

    def make_move(self, move, player):
        self.board = self.board[:move] + (player,) + self.board[move+1:]

    def imprimir_tablero(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        for i in range(0, 9, 3):
            print("|".join(symbols[x] for x in self.board[i:i+3]))
            print("-" * 6)

    def verificar_ganador(self):
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] != 0:
                return True
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] != 0:
                return True
        if self.board[0] == self.board[4] == self.board[8] != 0 or self.board[2] == self.board[4] == self.board[6] != 0:
            return True
        return False

    def turno_jugador(self, jugador):
        if jugador == 1:
            self.computer_move()
        else:
            self.agent_move()

    def start_game(self):
        jugadores = [0, 1]
        turno = 0

        while True:
            print("&"*20)
            self.imprimir_tablero()
            self.turno_jugador(jugadores[turno])
            if self.verificar_ganador():
                break
            elif 0 not in self.board:
                break
            turno = (turno + 1) % 2
        print("&"*20)
        self.imprimir_tablero()
        if 0 not in self.board:
            print("Empate!")
        elif turno == 1:
            print("El jugador aleatorio ganado!")
        elif turno == 0:
            print("El RL ha ganado!")

if __name__ == "__main__":
    game = Gato("PolicyIteration")
    game.start_game()