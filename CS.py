#Libraries

import random
import numpy as np 
from math import pi, sin, cos 
from scipy.special import gamma 
from matplotlib import pyplot as plt 
from copy import *
import pandas as pd


from random import seed
from math import *
import numpy as np
import pandas as pd

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

def fitness_value(function, vector, dimension):
    # Retorna apenas o valor do fitness (terceiro elemento da tupla)
    _, _, value = function(vector, dimension)
    return value
    

def limits(new_nest):
#Função que define os limites superiores e inferiores
    for i in range(len(new_nest)):
        for j in range(len(new_nest[i])):
            if new_nest[i][j] > superior_limit:
                new_nest[i][j] = random.uniform(inferior_limit,superior_limit)
            elif new_nest[i][j] < inferior_limit:
                new_nest[i][j] = random.uniform(inferior_limit,superior_limit)
    # Arredonda para inteiro após limitar
    return np.round(new_nest).astype(int)

def levy_flight(nest,gbest):
    u = np.random.normal(0.,sigma,np.shape(nest))
    v = np.random.normal(0.,1.,np.shape(nest))
    levy = (u/np.abs(v)**(1/beta))*(nest-gbest)
    alpha = 1.0
    nest = nest + alpha*levy
    nest = limits(nest)
    return nest

def random_flight(function,nest):
    fitness_array = np.zeros(len(nest))
    for i in range(len(nest)): 
        fitness_array[i] =  (fitness_value(function,nest[i],dimension))
    fitness_sort = fitness_array.copy() 
    fitness_sort.sort()
    discart = fitness_sort[-int(population*abandon_prob)]
    index = fitness_array > discart
    new_nest = np.random.uniform(inferior_limit,superior_limit,(population,dimension))
    nest[index] = np.random.uniform(inferior_limit,superior_limit,(population,dimension)) [index]
    new_nest = limits(nest)
    return new_nest

def new_gbest(function,nest,gbest,gbest_value):
    #avalia ninho a ninho, sendo o menor fitness guardado como gbest.

    for i in range(len(nest)):
        nest_fitness = fitness_value(function,nest[i],dimension)
        if nest_fitness < gbest_value:
            gbest = deepcopy(nest[i])
            gbest_value = nest_fitness
            
    
    return (gbest,gbest_value)


def cuckoo_min(function, local_interactions, local_population, local_inferior_limit, local_superior_limit, local_dimension, local_abandon_prob):
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

    nest = np.random.uniform(inferior_limit, superior_limit, (population, dimension))
    nest = np.round(nest).astype(int)  # Arredonda para inteiro ao criar
    gbest = nest[0]
    gbest_value = fitness_value(function, nest[0], dimension)
    gbest, gbest_value = new_gbest(function, nest, gbest, gbest_value)
    gbest_list = [gbest_value]
    counter = 0

    while counter < interactions:
        nest = levy_flight(nest, gbest)
        gbest, gbest_value = new_gbest(function, nest, gbest, gbest_value)
        nest = random_flight(function, nest)
        gbest, gbest_value = new_gbest(function, nest, gbest, gbest_value)
        gbest_list.append(gbest_value)
        counter += 1
        print('fitness =', gbest_value)
        print('Geração =', counter)

    # Retorna o melhor vetor também
    return gbest, gbest_value

def confiabilidade(vector,dimension):

    seed(118092648)
    ult = 23                # quantidade maxima de manutenções por componente
    j = 7                   # quantidade de componentes
    p = 0                   # probabilidade de manutenção não satisfatória
    tmis = 540              # tempo total da missão
    cmv = 20                # custo de manutenção da válvula
    cmb = 20                # custo de manutenção da bomba
    crv = 20                # custo de reparo da válvula
    crb = 20               # custo de reparo da bomba
    wd = 0.1
    wc = 0.0001/(ult*j)
    #print('vector antes =',vector)
    
    r = []                  # lista da confiabilidade final de cada componente
    rtodoscadatm = []       # guarda todas as confiabilidades calculadas de cada manutenção para cada componente
    custo_comp= []   
    confmediacomp = []

    #vector= [round(i) for i in vector]   #arredondar todos os elementos para o inteiro mais proximo

    lista_separada = []             #separar vector em uma lista de sete listas com 23 elementos cada
    for i in range(1, j+1):
        juntar = vector[(i-1)*ult:i*ult]
        lista_separada.append(juntar)

    lista_positivos = []            #deixa apenas os numeros posivos e menores que Tmis
    for i in range(j):
        lista_intermidiaria = []
        for k in range(ult):
            if lista_separada[i][k] > 0 and lista_separada[i][k] < tmis:
                lista_intermidiaria.append(lista_separada[i][k])
        if len(lista_intermidiaria) == 0:
            o = random.randint(1, tmis-1)
            lista_intermidiaria.append(o)
        lista_positivos.append(lista_intermidiaria)
    
    tm_final = []                  #tira os numeros repetidos
    for i in range(j):
        tm_intermediario = []
        for k in range(len(lista_positivos[i])):
            if lista_positivos[i][k] not in tm_intermediario:
                tm_intermediario.append(lista_positivos[i][k])
        tm_intermediario.sort()
        tm_final.append(tm_intermediario)

    for u in range(j):
        #definir parametros para cada componente

        #componentes 1 a 4 (indice 0 a 3) são valvulas
        #componentes 5 a 7 (indice 4 a 6) são bombas
        if u in range(0,4):     #componentes 1 2 3 e 4 são valvulas
            m = 1.35
            theta = 600
        else:                   #componentes 5, 6 e 7 são bombas
            m = 1.8
            theta = 1800
        
        #Calcular confiabilidade de cada manutenção para cada componente
        
        rman = []               # rman e a lista da confiabilidade de cada manutenção do componente em questão
        tam = len(tm_final[u])  #quantas manutenções estão agendadas para o componente
        soma = 0                #ajuda no calculo do somatorio da equação de probabilidade
        
        tm = tm_final[u]
        tult = tm[-1]
        t = random.randrange(int(tult), int(tmis))
        #t = tmis - tult
        
        #calcula a parte do somatório da confiabilidade
        for k in range(tam):    #k é a kesima manutenção do componente (primeira, segunda) não é em dias
            if k == 0:          #para a primeira manutenção
                somatorio = (-((tm[k])/theta)**m)
                soma = soma + somatorio
            else:
                somatorio = (-((tm[k]-tm[k-1])/theta)**m)
                soma = soma + somatorio

            #calcular o expoente que resulta na confiabilidade de cada manutenção e adicionar a resposta na lista rman
            expoente_cada_tm = -((t - tult)/theta)**m - p + soma
            #print (expoente_cada_tm)
            conf_cada_tm = exp(expoente_cada_tm)
            rman.append(conf_cada_tm)

        #confiabilidade da ultima manutenção agendada do componente
        rult = rman[-1]
        r.append(rult)

        #calcular a confiabilidade para o tmis
        somatorio = (-((tmis - tult)/theta)**m)
        soma = soma + somatorio
        expoentetmis = -((tmis - tult)/theta)**m -p *soma
        conf_tmis = exp(expoentetmis)

        #calcular o custo de cada manutencao para cada componente
        soma_custo = 0
        tam_custo = len(rman)

        #custo para a valvula 1 e 2
        if u == 0 or u ==1:
            for e in range(tam_custo):
                if e == 0:
                    somatorio_custo = (cmv+cmb)*rman[e] + crv*(1-rman[e])
                    soma_custo = soma_custo + somatorio_custo
                else:
                    somatorio_custo = (cmv+cmb)*(rman[e]/rman[e-1]) + crv*(1-(rman[e]/rman[e-1])) + crv*(1-(conf_tmis/rult))
                    soma_custo = soma_custo + somatorio_custo
            custo_comp.append(soma_custo)
            rtodoscadatm
        
        #custo para valvula 3 e 4
        if u == 2 or u == 3:
            for e in range(tam_custo):
                if e == 0:
                    somatorio_custo = (cmv + cmv + cmb)*rman[e] + (crv+crv)*(1-(rman[e]/rman[e-1])) + (crv+crv)*(1-(conf_tmis/rult))
                    soma_custo = soma_custo + somatorio_custo
                else:
                    somatorio_custo = (cmv+cmv+cmb)*(rman[e]/rman[e-1]) + (crv+crv)*(1-(conf_tmis/rult))
                    soma_custo = soma_custo + somatorio_custo
            custo_comp.append(soma_custo)
            rtodoscadatm.append(rman)

        #custo para bomba 1 e 3
        if u == 4 or u == 6:
            for e in range(tam_custo):
                if e == 0:
                    somatorio_custo = (cmv+cmb)*rman[e] + (crv+crb)*(1-(rman[e]/rman[e-1])) + (crv+crb)*(1-(conf_tmis/rult))
                    soma_custo = soma_custo + somatorio_custo
                else:
                    somatorio_custo = (cmv + cmb)*(rman[e]/rman[e-1])
                    soma_custo = soma_custo +somatorio_custo
            custo_comp.append(soma_custo)
            rtodoscadatm.append(rman)

        #custo para bomba 2
        if u == 5:
            for e in range(tam_custo):
                if e == 0:
                    somatorio_custo = (cmv+cmv+cmb)*rman[e] + (crv+crv+crb)*(1-rman[e])
                    soma_custo = soma_custo + somatorio_custo
                else:
                    somatorio_custo = (cmv+cmv+cmb)*(rman[e]/rman[e-1]) + (crv+crv+crb)*(1-(conf_tmis/rult))
                    soma_custo = soma_custo + somatorio_custo
            custo_comp.append(soma_custo)
            rtodoscadatm.append(rman)
        
        conf_media = 0
        intervalo_total = tult - tm[0]
        confiabilidade = []
        for w in range(1, tam):
            intervalo_dias = tm[w] - tm[w-1]
            divisao_por_dia = (rman[w-1]-rman[w])/intervalo_dias
            for b in range(intervalo_dias):
                confiabilidade_dia = rman[w-1]-(b*divisao_por_dia)
                confiabilidade.append(confiabilidade_dia)
                conf_media = conf_media + confiabilidade_dia
        if intervalo_total == 0:
            confiabilidademediacomp = conf_media
        else:
            confiabilidademediacomp = conf_media/intervalo_total
        confmediacomp.append(confiabilidademediacomp)
    
    #Confiabilidade média do sistema

    rmediaserie1 = (confmediacomp[4]) * (confmediacomp[0])                          #bomba 1 e valvula 1
    rmediaserie2 = (confmediacomp[5]) * (confmediacomp[1]) * (confmediacomp[2])     #bomba 2 e valvula 3 e 4
    rmediaserie3 = (confmediacomp[6]) * (confmediacomp[3])                          #bomba 3 e valvula 2
    rmediasistema = 1-((1-rmediaserie1) * (1-rmediaserie2) * (1-rmediaserie3))

    #Confiabilidade do sistema
    rserie1 = (r[4]) * (r[0])
    rserie2 = (r[5]) * (r[1]) * (r[2])
    rserie3 = (r[6]) * (r[3])
    rsistema = 1 - ((1-rserie1)*(1-rserie2)*(1-rserie3))

    custo_total = 0
    for y in range(len(custo_comp)):
        custo_total = custo_total + custo_comp[y]

    value = wd * (1 - rmediasistema) + wc * custo_total

    # Retorne também rmediasistema e custo_total
    return rmediasistema, custo_total, value


#cuckoo_min(confiabilidade,5000,100,-540,540,161,0.25)

def rodar_experimentos():
    resultados = []
    populacoes = [100]
    geracoes = [500, 1000, 2000]
    n_repeticoes = 5

    for pop in populacoes:
        for gen in geracoes:
            for rep in range(n_repeticoes):
                print(f"Rodando: População={pop}, Gerações={gen}, Repetição={rep+1}")
                gbest, best_value = cuckoo_min(
                    confiabilidade,
                    local_interactions=gen,
                    local_population=pop,
                    local_inferior_limit=-540,
                    local_superior_limit=540,
                    local_dimension=161,
                    local_abandon_prob=0.25
                )
                rmediasistema, custo_total, value = confiabilidade(gbest, 161)
                resultados.append({
                    "populacao": pop,
                    "geracoes": gen,
                    "repeticao": rep+1,
                    "rmediasistema": rmediasistema,
                    "custo_total": custo_total,
                    "value": value,
                    "vetor_ninho": list(gbest)
                })

    df = pd.DataFrame(resultados)
    df.to_excel("resultados_experimentos.xlsx", index=False)
    print("Todos os resultados foram salvos em 'resultados_experimentos.xlsx'.")

if __name__ == "__main__":
    rodar_experimentos()