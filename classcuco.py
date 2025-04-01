import random
import numpy as np 
from math import pi, sin, cos 
from scipy.special import gamma 
from matplotlib import pyplot as plt 
import time  # Importa o módulo time


class Graph_controller(object):
    def __init__(self):
        pass
    
    @staticmethod
    def plot(fitness1, fitness2):
        plt.scatter(fitness1, fitness2)
        plt.xlabel("f1(x)")
        plt.ylabel("f2(x)")
        plt.title("Fronteira de Pareto")
        plt.savefig("pareto_plot.png")
        
class Fitness_controller(object):
    @staticmethod
    def zdt1(vector, dimension):
        f1 = vector[0]
        g = 1.0 + 9.0 * np.sum(vector[1:]) / (dimension - 1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([f1, f2])
    
    @staticmethod
    def sch(vector):
        """Schaffer's Min-Min (SCH) function (convex Pareto front)."""
        f1 = vector[0]**2
        f2 = (vector[0] - 2)**2
        return np.array([f1, f2])



    @staticmethod
    def zdt2(vector, dimension=30):
        """ZDT2 function (non-convex Pareto front)."""
        f1 = vector[0]
        g = 1.0 + 9.0 * np.sum(vector[1:]) / (dimension - 1)
        h = 1.0 - (f1 / g)**2  # Difers from ZDT1 here
        f2 = g * h
        return np.array([f1, f2])

    @staticmethod
    def zdt3(vector, dimension=30):
        """ZDT3 function (discontinuous Pareto front)."""
        f1 = vector[0]
        g = 1.0 + 9.0 * np.sum(vector[1:]) / (dimension - 1)
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)  # Additional sinusoidal term
        f2 = g * h
        return np.array([f1, f2])

    @staticmethod
    def lz(vector, dimension=10):
        """LZ function (complex Pareto set)."""
        f1 = vector[0] + (2.0 / len(vector[1::2])) * np.sum(
            (vector[1::2] - np.sin(6 * np.pi * vector[0] + (np.arange(2, dimension + 1, 2) * np.pi / dimension))**2
        ))
        f2 = 1 - np.sqrt(vector[0]) + (2.0 / len(vector[2::2])) * np.sum(
            (vector[2::2] - np.sin(6 * np.pi * vector[0] + (np.arange(3, dimension + 1, 2) * np.pi / dimension))**2
        ))
        return np.array([f1, f2])

    @staticmethod
    def custom_function(vector,dimension):
        """
        Função personalizada para minimizar:
        f1(x, y) = 4x^2 + 4y^2
        f2(x, y) = (x - 5)^2 + (y - 5)^2
        """
        x, y = vector[0], vector[1]
        f1 = 4 * x**2 + 4 * y**2
        f2 = (x - 5)**2 + (y - 5)**2
        return np.array([f1, f2])

class Environment:
    """
    Classe para encapsular as variáveis globais compartilhadas entre as classes.
    """
    def __init__(self, dimension, lower_bound, upper_bound, fitness_function):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fitness_function = fitness_function


class Egg:
    def __init__(self, env: Environment):
        self.env = env
        self.pareto = True
        self.function_parameters = self.create_function_parameters()
        self.fitness = self.calculate_fitness()

    def create_function_parameters(self):
        return np.random.uniform(self.env.lower_bound, self.env.upper_bound, self.env.dimension)

    def calculate_fitness(self):
        return self.env.fitness_function(self.function_parameters, self.env.dimension)


class Nest:
    def __init__(self, eggs_number, env: Environment):
        self.eggs_number = eggs_number
        self.env = env
        self.eggs = self.create_eggs()
        self.gbest = self.calculate_gbest()

    def create_eggs(self):
        return [Egg(self.env) for _ in range(self.eggs_number)]

    def calculate_gbest(self):
        gbest = None
        for egg in self.eggs:
            if egg.pareto:
                gbest = egg
                break
        if gbest is None:
            gbest = random.choice(self.eggs)
        return gbest

    def levy_flight(self):
        self.gbest = self.calculate_gbest()
        if self.gbest is None:
            self.gbest = random.choice(self.eggs)  # Fallback para um ovo aleatório

        alpha = 1.0
        beta = 1.5
        sigma = (gamma(1 + beta) * sin(pi * beta / 2.0) /
                 (gamma((1 + beta) / 2) * beta * 2.0 ** ((beta - 1) / 2.0))) ** (1.0 / beta)

        for egg in self.eggs:
            u = np.random.normal(0., sigma, np.shape(egg.function_parameters))
            v = np.random.normal(0., 1., np.shape(egg.function_parameters))
            levy_step = (u / np.abs(v) ** (1 / beta)) * (self.gbest.function_parameters - egg.function_parameters)
            egg.function_parameters += alpha * levy_step
            egg.function_parameters = np.clip(egg.function_parameters, self.env.lower_bound, self.env.upper_bound)
            egg.fitness = egg.calculate_fitness()


class Farmer:
    def __init__(self, population_size, eggs_number, abandon_rate, env: Environment):
        self.env = env
        self.nests = self.create_nests(population_size, eggs_number)
        self.pareto_optimal = []
        self.abandon_rate = abandon_rate

    def create_nests(self, population_size, eggs_number):
        return [Nest(eggs_number, self.env) for _ in range(population_size)]

    def not_dominated(self, egg1, egg2):
        """
        Verifica se egg1 não é dominado por egg2.
        """
        return np.all(egg1.fitness <= egg2.fitness) and np.any(egg1.fitness < egg2.fitness)

    def check_pareto(self):
        """
        Marca os ovos como Pareto ótimos ou não, com base na dominância.
        """
        # Inicialmente, marca todos os ovos como Pareto ótimos
        for nest in self.nests:
            for egg in nest.eggs:
                egg.pareto = True

        # Verifica se algum ovo é dominado por outro
        for nest in self.nests:
            for egg in nest.eggs:
                for other_nest in self.nests:
                    for other_egg in other_nest.eggs:
                        if egg != other_egg and self.not_dominated(other_egg, egg):
                            egg.pareto = False
                            break

    def create_pareto_optimal(self):
        """
        Cria uma lista de ovos Pareto ótimos.
        """
        pareto_optimal = []
        for nest in self.nests:
            for egg in nest.eggs:
                if egg.pareto:
                    pareto_optimal.append((egg.function_parameters, egg.fitness))
        self.pareto_optimal = pareto_optimal

    def apply_levy_flight(self):
        for nest in self.nests:
            nest.levy_flight()

    def abandon_nests(self):
        """
        Abandona os ninhos com base no abandon_rate e cria novos ninhos para substituí-los.
        """
        # Ordena os ninhos com base no número de ovos Pareto ótimos (maior para menor)
        self.nests.sort(key=lambda nest: sum(egg.pareto for egg in nest.eggs), reverse=True)

        # Calcula o número de ninhos a serem abandonados
        num_to_abandon = int(len(self.nests) * self.abandon_rate)

        # Remove os piores ninhos (os que estão mais à direita na lista)
        for _ in range(num_to_abandon):
            self.nests.pop()

        # Cria novos ninhos para substituir os removidos
        for _ in range(num_to_abandon):
            new_nest = Nest(self.nests[0].eggs_number, self.env)  # Usa os mesmos parâmetros dos ninhos existentes
            self.nests.append(new_nest)


def main(max_generation, population_size, eggs_number, abandon_rate, dimension, lower_bound, upper_bound, fitness_function):
    start_time = time.time()  # Marca o início do tempo

    env = Environment(dimension, lower_bound, upper_bound, fitness_function)
    alegrio = Farmer(population_size, eggs_number, abandon_rate, env)

    for i in range(max_generation):
        alegrio.check_pareto()
        alegrio.create_pareto_optimal()  # Atualiza a lista de Pareto ótimos
        alegrio.apply_levy_flight()
        alegrio.check_pareto()
        alegrio.create_pareto_optimal()  # Atualiza novamente após o voo de Lévy
        alegrio.abandon_nests()
        # print (alegrio.pareto_optimal)
        print(f"Geração: {i + 1}/{max_generation} | Pareto ótimos: {len(alegrio.pareto_optimal)}")

    if len(alegrio.pareto_optimal) > 0:
        x = [item[0][0] for item in alegrio.pareto_optimal]
        y = [item[1][1] for item in alegrio.pareto_optimal]
        Graph_controller.plot(x, y)
    else:
        print("Not enough Pareto optimal solutions to plot.")

    end_time = time.time()  # Marca o final do tempo
    elapsed_time = end_time - start_time  # Calcula o tempo decorrido
    print(f"Tempo total de execução: {elapsed_time:.2f} segundos")  # Exibe o tempo total


main(max_generation=250,
     population_size=100,
     eggs_number=2,
     abandon_rate=0.25,
     dimension=2,
     lower_bound=0,
     upper_bound=5,
     fitness_function=Fitness_controller.custom_function)