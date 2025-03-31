import random
import numpy as np 
from math import pi, sin, cos 
from scipy.special import gamma 
from matplotlib import pyplot as plt 
from copy import *
class Graph_controller(object):
    def __init__(self):
        pass
    
    @staticmethod
    def plot(fitness1, fitness2):
        plt.scatter(fitness1, fitness2)
        plt.show()
        
class Fitness_controller(object):
    def __init__(self, dimension):
        self.dimension = dimension
        pass
    
    def zdt1(vector,dimension):
        f1 = vector[0]
        g = 1 + 9 * np.sum(vector[1:]) / (len(vector) - 1) if len(vector) > 1 else 1
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])

class Farmer(object):
    def __init__(self, population_size, eggs_number, dimension, lower_bound, upper_bound, fitness_function):
        self.nests = self.create_nests(population_size, eggs_number, dimension, lower_bound, upper_bound,fitness_function)
        self.check_pareto()
        self.pareto_optimal = self.create_pareto_optimal()
       
    def create_nests(self, population_size, eggs_number, dimension, lower_bound, upper_bound,fitness_function):
        """
        Essa função cria um numero de ninhos igual ao population_size. Cada ninho é composto um objeto da classe Ninho"""
            
        self.nests = []
        for i in range(population_size):
            nest = Ninho(eggs_number, dimension, lower_bound, upper_bound, fitness_function)
            self.nests.append(nest)
        return self.nests
    
    def check_pareto(self):
        """
        Essa função verifica se o ovo é pareto ótimo. Para isso, ela compara o ovo com todos os outros ovos de todos os ninhos.
        Se o ovo for dominado por outro ovo, ele não é pareto ótimo.
        """
        for nest in self.nests:
            for egg in nest.eggs:
                
                for other_nest in self.nests:
                    for other_egg in other_nest.eggs:
                        if egg != other_egg and self.not_dominated(other_egg, egg):
                            egg.pareto = False
                            break

    def not_dominated(self, egg1, egg2):
        '''
        Essa função verifica se o ovo 1 tem pelo menos um valor de fitness maior que o ovo 2.
        Se sim, o ovo 1 não é dominado pelo ovo 2.
        '''
        return not (np.any(egg1.fitness < egg2.fitness) )

    
    def create_pareto_optimal(self):
        """
        Essa função percorre todos os ovos do ninho e verifica quais deles são ovos pareto ótimo.
        Ao encontrar um ovo pareto ótimo, ele o guarda na lista de ovos pareto ótimo.
        """
        pareto_optimal = []
        for nest in self.nests:
            for egg in nest.eggs:
                print("egg", egg.function_parameters, egg.pareto)
                if egg.pareto:
                    pareto_optimal.append((egg.function_parameters, egg.fitness))
                    

        return pareto_optimal
    
    def remove_not_pareto_optimal(self):
        """
        Essa função percorre todos os ovos ditos pareto otimos pela variavel self.pareto_optimal e verifica se eles realmente são pareto otimos.
        Se algum deles não for, ele o remove da lista de ovos pareto ótimo.
        """
        for egg in self.pareto_optimal:
            if not self.check_pareto(egg):
                self.pareto_optimal.remove(egg)
        return self.pareto_optimal
    

class Egg(object):
    def __init__(self,dimension,lower_bound, upper_bound,fitness_function):

        self.pareto = True
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fitness_function = fitness_function
        self.function_parameters = self.create_function_parameters()
        self.fitness = self.calculate_fitness()

    def create_function_parameters(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.dimension))
    
    def calculate_fitness(self):
        """
        Essa função calcula o valor de fitness do ovo, que é a soma dos quadrados dos parâmetros da função.
        """
        return self.fitness_function(self.function_parameters, self.dimension)
            
class Ninho(object):
    def __init__(self,eggs_number, dimension, lower_bound, upper_bound, fitness_function):
        self.eggs_number = eggs_number
        self.eggs = self.create_eggs(dimension, lower_bound, upper_bound, fitness_function)
        self.gbest = self.calculate_gbest()

    def create_eggs(self,dimension, lower_bound, upper_bound, fitness_function):
        eggs = []
       
        for i in range(self.eggs_number):
            egg = Egg(dimension, lower_bound, upper_bound, fitness_function)
            eggs.append(egg)
        
        return eggs

    def calculate_gbest(self):      
        """
        Essa função percorre todos os ovos do ninho e verifica se algum deles é um ovo pareto ótimo.
        Se encontrar um ovo pareto ótimo, ele o define como gbest. Caso contrário, escolhe um ovo aleatório do ninho.
        """
        gbest = None
        for egg in self.eggs:
            if egg.pareto:
                gbest = egg
                break        
        if gbest is None:
            gbest = random.choice(self.eggs)

    # def levy_flight(self):
    #     alpha = 1.0
    #     beta = 1.5
    #     sigma = (gamma(1+beta)*sin(pi*beta/2.0)/(gamma((1+beta)/2)*beta*2.0**((beta-1)/2.0)))**(1.0/beta)
    #     u = np.random.normal(0.,sigma,np.shape(nest))
    #     v = np.random.normal(0.,1.,np.shape(nest))
    #     levy = (u/np.abs(v)**(1/beta))*(self.gbest-nest)
    #     nest = nest + alpha*levy
    #     nest = limits(nest)
    #     return nest


def main(population_size, eggs_number, dimension, lower_bound, upper_bound, fitness_function):
    joao = Farmer(population_size, eggs_number, dimension, lower_bound, upper_bound, fitness_function)
    # print("len ninhos", len(joao.nests))
    # for i in range(len(joao.nests)):
    #     print("len ovos", len(joao.nests[i].eggs))
    if len(joao.pareto_optimal) > 0:
        x = [item[0][0] for item in joao.pareto_optimal]
        y = [item[1][1] for item in joao.pareto_optimal]
        Graph_controller.plot(x, y)
    else:
        print("Not enough pareto optimal solutions to plot.")



main(10, 2, 2, 0, 10, Fitness_controller.zdt1)