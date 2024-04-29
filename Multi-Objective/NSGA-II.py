import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the Individual class
class Individual:
    def __init__(self, point, problem, objectives=None):
        self.point = point  # The point in the problem space
        self.Sp = set()  # Set of solutions dominated by this solution
        self.N = 0  # Number of solutions that dominate this solution
        self.rank = None  # Rank of the solution
        self.distance = 0  # Crowding distance
        self.values = problem.evaluate(point)  # Objective function values

# Define the NSGA-II algorithm
class NSGA2:
    def __init__(self, generations, population_size, mutaition_rate, problem):
        self.population_size = population_size  # Population size
        self.mutation_rate = mutaition_rate  # Mutation rate
        self.problem = problem  # Problem to solve
        self.n_var = self.problem.n_var  # Number of variables
        self.n_obj = self.problem.n_obj  # Number of objectives
        self.run(generations)  # Run the algorithm for a certain number of generations
        
    # Generate the initial population depending on the problem
    def generate_population(self, n):
        if isinstance(self.problem, WFG1):
            return [Individual(np.array([np.random.uniform(0, (i+1)*2) for i in range(self.n_var)]), self.problem) for _ in range(n)]
        elif isinstance(self.problem, MO_LANDER) or isinstance(self.problem, DeepSeaTreasure) or isinstance(self.problem, MountainCar):
            # Obtener los parámetros de la red como una lista de tensores
            params = [w.detach().numpy() for w in self.problem.net.parameters()]
            
            # Aplanar cada tensor en la lista
            flattened_params = [p.flatten() for p in params]
            
            # Concatenar todos los tensores aplanados en un solo tensor
            concatenated_params = np.concatenate(flattened_params)

            params = [concatenated_params for _ in range(n)]
            
            return [Individual(p, self.problem) for p in params]
        else:
            return [Individual(np.random.uniform(0, 1, self.n_var), self.problem) for _ in range(n)]

    # Check if solution p dominates solution q
    def dominates(self, p, q):
        return all(p_i <= q_i for p_i, q_i in zip(p, q)) and any(p_i < q_i for p_i, q_i in zip(p, q))

    # Perform non-dominated sorting
    def non_dominated_sort(self, individuals):

        F = defaultdict(list)
        for p in individuals:
            for q in individuals:
                if p == q:
                    continue
                if self.dominates(p.values, q.values):
                    p.Sp.add(q)
                elif self.dominates(q.values, p.values):
                    p.N += 1
            if p.N == 0:
                p.rank = 1
                F[1].append(p)
        i = 1
        while F[i]:
            Q = []
            for p in F[i]:
                for q in p.Sp:
                    q.N -= 1
                    if q.N == 0:
                        q.rank = i + 1
                        Q.append(q)
            i += 1
            F[i] = Q

        return F
    
    # =======================Crowding Distance===================
    def sort_by_objective(self, front, m):
        return sorted(front, key=lambda p: p.values[m])

    def crowding_distance_assignment(self, I):
        l = len(I)
        if l == 0:
            return
        for m in range(I[0].values.size):
            I = self.sort_by_objective(I, m)
            f_min = I[0].values[m]
            f_max = I[-1].values[m]
            I[0].distance = float('inf')
            I[-1].distance = float('inf')
            for i in range(1, l - 1):
                I[i].distance += (I[i + 1].values[m] - I[i - 1].values[m])/(f_max - f_min + 1e-8)
        
    
        
    def sort_by_crowed_comparation(self, front):
        return sorted(front, key=lambda p: (-p.rank, p.distance), reverse=True)
    
    # =============================GA=============================

    # Perform binary tournament selection
    def binary_tournament_selection(self, P):
        winners = []
        while len(winners) < len(P)//2:
            vs = np.random.choice(P, 2)
            winners.append(max(vs, key=lambda p: (-p.rank, p.distance)))
        return winners

    # Perform crossover
    def crossover(self, individual1, individual2):
        
        if individual1.point.size == 1:
            individual1 = individual1.point[0]
            individual2 = individual2.point[0]
            new_individual1 = Individual(np.array([0.2 * (individual1 + individual2)]), self.problem)
            new_individual2 = Individual(np.array([0.8 * (individual1 + individual2)]), self.problem)
        else:
            point1 = individual1.point.copy()
            point2 = individual2.point.copy()

            # Elegir un índice de cruce al azar
            crossover_index = np.random.randint(0, len(point1))

            # Intercambiar un elemento de los puntos
            point1[crossover_index], point2[crossover_index] = point2[crossover_index], point1[crossover_index]

            
            new_individual1 = Individual(point1, self.problem)
            new_individual2 = Individual(point2, self.problem)

        return new_individual1, new_individual2

    # Perform mutation
    def mutate(self, individual, mutation_rate):
        point = individual.point.copy()

        # Recorrer cada elemento en el punto
        for i in range(len(point)):
            # Aplicar la mutación con una probabilidad igual a mutation_rate
            if random.random() < mutation_rate:
                # Añadir un pequeño número aleatorio al elemento
                point[i] += random.uniform(-0.1, 0.1)
                if isinstance(self.problem, WFG1):
                    point[i] = max(0, min(2, point[i]))
                elif isinstance(self.problem, Problem):
                    point[i] = max(-1, min(1, point[i]))
                else:
                    point[i] = max(0, min(1, point[i]))
            

        # Crear un nuevo individuo con el punto mutado
        point = np.array(point)
        new_individual = Individual(point, self.problem)
        return new_individual
    
    # Generate offspring
    def generate_offspring(self, P, mutation_rate):
        offspring = []
        winners = self.binary_tournament_selection(P)
        while len(offspring) < len(P):
            parents = np.random.choice(winners, 2)
            offspring += self.crossover(*parents)
            offspring = [self.mutate(p, mutation_rate) for p in offspring]
        return offspring
    
    # Initialize the population
    def initialize(self):
        self.P_t = self.generate_population(self.population_size)
        F = self.non_dominated_sort(self.P_t)
        for i in range(1, len(F)):
            self.crowding_distance_assignment(F[i])
        
        self.Q_t = self.generate_offspring(self.P_t, self.mutation_rate)
        
    # Run the algorithm for a certain number of generations
    def run(self, generations):
        self.initialize()

        # Inicializar las listas de recompensas
        self.rewards = [[] for _ in range(self.n_obj)]

        for t in tqdm(range(generations)):
            self.R_t = self.P_t + self.Q_t
            self.F = self.non_dominated_sort(self.R_t)
            self.P_t = []
            i = 0
            while len(self.P_t) + len(self.F[i]) <= self.population_size:
                self.crowding_distance_assignment(self.F[i])
                self.P_t += self.F[i]
                i += 1
            self.sort_by_crowed_comparation(self.F[i])
            self.P_t += self.F[i][:self.population_size - len(self.P_t)]
            self.Q_t = self.generate_offspring(self.P_t, self.mutation_rate)

            # Calcular y almacenar las recompensas promedio para cada objetivo
            for i in range(self.n_obj):
                avg_reward = np.mean([individual.values[i] for individual in self.P_t])
                self.rewards[i].append(avg_reward)

        # if self.n_obj == 2:
        #     self.plot_pareto_front()
        # elif self.n_obj == 3:
        #     self.plot_pareto_front_3d()

    def plot_pareto_front(self):
        x = [individual.values[0] for individual in self.P_t]
        y = [individual.values[1] for individual in self.P_t]

        plt.scatter(x, y)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front')
        plt.show()
    
    def plot_pareto_front_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = [individual.values[0] for individual in self.P_t]
        y = [individual.values[1] for individual in self.P_t]
        z = [individual.values[2] for individual in self.P_t]
        ax.scatter(x, y, z)
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        plt.show()