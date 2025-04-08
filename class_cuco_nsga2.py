import random
import numpy as np
from math import pi, sin, cos
from scipy.special import gamma
from matplotlib import pyplot as plt
import time
import copy
import pandas as pd

class Graph_controller(object):
    @staticmethod
    def plot_zdt1(pareto_optimal):
        fitness1 = [egg.fitness[0] for egg in pareto_optimal]
        fitness2 = [egg.fitness[1] for egg in pareto_optimal]

        x_true = np.linspace(0, 1, 500)
        f1_true = x_true
        f2_true = 1 - np.sqrt(x_true)
        plt.plot(f1_true, f2_true, 'r-', linewidth=2, label="Fronteira de Pareto Verdadeira (ZDT1)", zorder=1)

        plt.scatter(fitness1, fitness2, s=10, color="blue", label="Soluções Calculadas", zorder=2)
        plt.xlabel("f1(x)")
        plt.ylabel("f2(x)")
        plt.title("Fronteira de Pareto - Comparação com ZDT1")
        plt.legend()
        plt.grid(True)
        plt.savefig("pareto_plot_zdt1.png")
        plt.show()

class Fitness_controller(object):
    @staticmethod
    def zdt1(vector, dimension):
        f1 = vector[0]
        g = 1.0 + ((9.0 * np.sum(vector[1:])) / (dimension - 1))
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([f1, f2])

class Environment:
    def __init__(self, dimension, lower_bound, upper_bound, fitness_function):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fitness_function = fitness_function

class Egg:
    def __init__(self, env: Environment):
        self.env = env
        self.pareto = True
        self.rank = None
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
        self.rank_ninho = None
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

        alpha = 1.0
        beta = 1.5
        sigma = (gamma(1 + beta) * sin(pi * beta / 2.0) /
                 (gamma((1 + beta) / 2) * beta * 2.0 ** ((beta - 1) / 2.0))) ** (1.0 / beta)

        for egg in self.eggs:
            u = np.random.normal(0., sigma, np.shape(egg.function_parameters))
            v = np.random.normal(0., 1., np.shape(egg.function_parameters))
            levy_step = (u / np.abs(v) ** (1 / beta)) * (self.gbest.function_parameters - egg.function_parameters)
            egg.function_parameters += alpha * levy_step

            for i in range(len(egg.function_parameters)):
                if egg.function_parameters[i] < self.env.lower_bound or egg.function_parameters[i] > self.env.upper_bound:
                    egg.function_parameters[i] = np.random.uniform(self.env.lower_bound, self.env.upper_bound)

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
        return np.all(egg1.fitness <= egg2.fitness) and np.any(egg1.fitness < egg2.fitness)

    def check_pareto(self):
        for nest in self.nests:
            for egg in nest.eggs:
                egg.pareto = True

        for nest in self.nests:
            for egg in nest.eggs:
                for other_nest in self.nests:
                    for other_egg in other_nest.eggs:
                        if egg != other_egg and self.not_dominated(other_egg, egg):
                            egg.pareto = False
                            break

        for nest in self.nests:
            for egg in nest.eggs:
                if egg.pareto:
                    self.pareto_optimal.append(copy.deepcopy(egg))

        non_dominated = []
        for egg in self.pareto_optimal:
            is_dominated = False
            for other_egg in self.pareto_optimal:
                if egg != other_egg and self.not_dominated(other_egg, egg):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(egg)
        self.pareto_optimal = non_dominated

        unique_pareto = []
        seen = set()
        for egg in self.pareto_optimal:
            egg_signature = (tuple(egg.fitness), tuple(egg.function_parameters))
            if egg_signature not in seen:
                seen.add(egg_signature)
                unique_pareto.append(egg)
        self.pareto_optimal = unique_pareto

    def rank_eggs(self):
        for nest in self.nests:
            for egg in nest.eggs:
                egg.rank = None

        current_rank = 0
        remaining_eggs = [egg for nest in self.nests for egg in nest.eggs]

        while remaining_eggs:
            non_dominated = []
            for egg in remaining_eggs:
                is_dominated = False
                for other_egg in remaining_eggs:
                    if egg != other_egg and self.not_dominated(other_egg, egg):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(egg)

            for egg in non_dominated:
                egg.rank = current_rank

            remaining_eggs = [egg for egg in remaining_eggs if egg not in non_dominated]
            current_rank += 1

    def apply_levy_flight(self):
        for nest in self.nests:
            nest.levy_flight()

    def abandon_nests(self):
        self.nests.sort(key=lambda nest: nest.rank_ninho)
        num_to_abandon = int(len(self.nests) * self.abandon_rate)

        for _ in range(num_to_abandon):
            self.nests.pop()

        for _ in range(num_to_abandon):
            new_nest = Nest(self.nests[0].eggs_number, self.env)
            self.nests.append(new_nest)

    def save_pareto_to_excel(self, filename="pareto_optimal.xlsx"):
        data = []
        for egg in self.pareto_optimal:
            row = list(egg.fitness) + list(egg.function_parameters)
            data.append(row)

        num_params = len(self.pareto_optimal[0].function_parameters) if self.pareto_optimal else 0
        columns = ["f1", "f2"] + [f"x{i+1}" for i in range(num_params)]
        df = pd.DataFrame(data, columns=columns)
        df.to_excel(filename, index=False)

def main(max_generation, population_size, eggs_number, abandon_rate, dimension, lower_bound, upper_bound, fitness_function):
    env = Environment(dimension, lower_bound, upper_bound, fitness_function)
    alegrio = Farmer(population_size, eggs_number, abandon_rate, env)

    for _ in range(max_generation):
        alegrio.check_pareto()
        alegrio.rank_eggs()
        alegrio.apply_levy_flight()
        alegrio.check_pareto()
        alegrio.rank_eggs()
        alegrio.abandon_nests()

    if len(alegrio.pareto_optimal) > 0:
        alegrio.save_pareto_to_excel()
        Graph_controller.plot_zdt1(alegrio.pareto_optimal)

main(max_generation=2,
     population_size=2,
     eggs_number=2,
     abandon_rate=0.5,
     dimension=30,
     lower_bound=0,
     upper_bound=1,
     fitness_function=Fitness_controller.zdt1)