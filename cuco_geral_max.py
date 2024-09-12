#Libraries

import random
import numpy as np 
from math import pi, sin, cos 
from scipy.special import gamma 
from matplotlib import pyplot as plt 
from copy import *

#Variaveis globais

global gbest_value
global gbest_list
global interactions
global inferior_limit
global superior_limit
global population
global dimension
global abandon_prob
global beta
global sigma



gbest_list = [0]
interactions = 0
inferior_limit = 0
superior_limit = 0
population = 0
dimension = 0
abandon_prob = 0

beta = 1.5
sigma = (gamma(1+beta)*sin(pi*beta/2.0)/(gamma((1+beta)/2)*beta*2.0**((beta-1)/2.0)))**(1.0/beta)

#Fitness -> Mudando está função, se aplica a qualquer cenário

def fitness_value(function,vector,dimension):
   
    return function(vector,dimension)
    

def limits(new_nest):
#Função que define os limites superiores e inferiores
    for i in range(len(new_nest)):
        for j in range(len(new_nest[i])):
            if new_nest[i][j] > superior_limit:
                new_nest[i][j] = random.uniform(inferior_limit,superior_limit) #Nesse caso, o random.uniform gera um float de varias casas. Dependendo do problema vale a pena mudar
            elif new_nest[i][j] < inferior_limit:
                new_nest[i][j] = random.uniform(inferior_limit,superior_limit)
    
    return (new_nest)

def levy_flight(nest,gbest):
    u = np.random.normal(0.,sigma,np.shape(nest))
    v = np.random.normal(0.,1.,np.shape(nest))
    levy = (u/np.abs(v)**(1/beta))*(nest-gbest)
    alpha = 1.0
    nest = nest + alpha*levy
    nest = limits(nest)
    return nest

def random_flight(function,nest):
    fitness_array = np.zeros(len(nest)) #cria um array completo de zeros do tamanho do ninho (numero de dimensoes)
    for i in range(len(nest)): 
        fitness_array[i] =  (fitness_value(function,nest[i],dimension)) #avalia cada ninho
    fitness_sort = fitness_array.copy() 
    fitness_sort.sort() #com a avaliação de cada ninho, ordena os resultados

    discart = fitness_sort[-int(population*abandon_prob)] #descarta os resultados maiores
    index = fitness_array > discart

    new_nest = np.random.uniform(inferior_limit,superior_limit,(population,dimension))
    nest[index] = np.random.uniform(inferior_limit,superior_limit,(population,dimension)) [index]
    new_nest = limits(nest) #confere as variaveis se estão dentro dos limites

    return new_nest

def new_gbest(function,nest,gbest,gbest_value):
    #avalia ninho a ninho, sendo o menor fitness guardado como gbest.

    for i in range(len(nest)):
        nest_fitness = fitness_value(function,nest[i],dimension)
        if nest_fitness > gbest_value:
            gbest = deepcopy(nest[i])
            gbest_value = nest_fitness
            
    
    return (gbest,gbest_value)


def cuckoo_min(function,local_interactions,local_population,local_inferior_limit,local_superior_limit,local_dimension,local_abandon_prob):
    
    global gbest_value
    global gbest_list
    global interactions
    global inferior_limit
    global superior_limit
    global population
    global dimension
    global abandon_prob
    global beta
    global sigma
        
    interactions = local_interactions
    inferior_limit = local_inferior_limit
    superior_limit = local_superior_limit
    population = local_population
    dimension = local_dimension
    abandon_prob = local_abandon_prob

    nest = np.random.uniform(inferior_limit,superior_limit,(population,dimension)) #cria os ninhos
    gbest = nest[0] #cria a variavel gbest colocando o primeiro ninho
    gbest_value = fitness_value(function,nest[0],dimension)
    gbest,gbest_value = new_gbest(function,nest,gbest,gbest_value) #avalia qual é o melhor fitness dentre os ninhos gerados e o guarda as variaveis do ninho
    gbest_list = [gbest_value] #cria a variavel que listará o gbest de cada geração afim de fazer um grafico
    counter = 0 #contador

    while counter < interactions:
        nest = levy_flight(nest, gbest) #realiza o voo de levy para cada ninho
        gbest,gbest_value = new_gbest(function,nest,gbest,gbest_value) #avalia o novo melhor fitness
        nest = random_flight(function,nest) #realiza o voo aleatorio
        gbest,gbest_value = new_gbest(function,nest,gbest,gbest_value) #avalia um novo gbest e guarda as variaveis do melhor ninho
        gbest_list.append(gbest_value) #guarda o novo gbest desta geração
        counter += 1
        print ('fitness =',gbest_value)
        print ('Geração =', counter)
    #print (fitness_value(function,gbest,dimension),gbest)
    
    interaction_number = list(range(interactions+1))
    
    plt.plot(interaction_number,gbest_list,marker='.')
    plt.show()
    return (gbest_value)
    

