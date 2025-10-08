from cuco_geral_min import *
#from cuco_geral_max import *
from random import seed
from math import *
import numpy as np
import pandas as pd

#funções

def sphere(vector,dimension):
    vector = np.array(vector)
    vector = vector**2
    return sum(vector)

def rosenbrock(vector,dimension):
    vector = np.array(vector)
    value = 3905.93-(100*(((vector[0]**2)-(vector[1]))**2)+(1-vector[0])**2)
    return value

def himmelblau(vector,dimension):
    vector = np.array(vector)
    value = (((vector[0]**2)+vector[1]-11)**2) + ((vector[0]+(vector[1]**2)-7)**2)
    return value

def schaffer(vector,dimension):
    vector = np.array(vector)
    value = 0.5 + ((sin(sqrt((vector[0]**2)+(vector[1]**2)))**2 - 0.5) / ((1+0.001*(vector[0]**2 + vector[1]**2))**2))
    return value
    
def rastringin(vector,dimension):
    vector = np.array(vector)
    resultados = np.zeros(dimension)
    for i in range(dimension):
        resultados[i] = vector[i]**2 -10*cos(2*3.14*vector[i])+10 
        
   
    return sum(resultados)
    


def confiabilidade(vector,dimension):

    seed(118092648)
    ult = 23                # quantidade maxima de manutenções por componente
    j = 7                   # quantidade de componentes
    p = 0                   # probabilidade de manutenção não satisfatória
    tmis = 540              # tempo total da missão
    cmv = 15                # custo de manutenção da válvula
    cmb = 80                # custo de manutenção da bomba
    crv = 40                # custo de reparo da válvula
    crb = 360               # custo de reparo da bomba
    wd = 0.1
    wc = 0.0001/(ult*j)
    #print('vector antes =',vector)
    
    r = []                  # lista da confiabilidade final de cada componente
    rtodoscadatm = []       # guarda todas as confiabilidades calculadas de cada manutenção para cada componente
    custo_comp= []   
    confmediacomp = []

    vector= [round(i) for i in vector]   #arredondar todos os elementos para o inteiro mais proximo

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
        tult= tm[-1]
        t = random.randrange(tult,tmis)
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

    value = wd * (1-rmediasistema) + wc*custo_total
    #print('Melhor solução =', tm_final)
    print('Custo =', custo_total)
    #print('Não conf =', wd*(1-rmediasistema))
    print('Risco = ', 1-rmediasistema)
    print('fitness =', value)
    
    #print('vector depois =',vector)
    return value



def iterar_funcao(n, funcao, *args, **kwargs):
    """
    Executa a função especificada n vezes e armazena os resultados em um DataFrame.
    
    Args:
        n (int): O número de iterações.
        funcao (function): A função a ser iterada.
        *args: Argumentos posicionais a serem passados para a função.
        **kwargs: Argumentos de palavras-chave a serem passados para a função.
        
    Returns:
        pandas.DataFrame: DataFrame contendo os resultados de cada iteração.
    """
    resultados = []
    for i in range(n):
        resultado_iteracao = funcao(*args, **kwargs)
        resultados.append(resultado_iteracao)
    
    # Convertendo os resultados em DataFrame
    df = pd.DataFrame(resultados)
    return df


#cuckoo_min(funcao,interações,população,limite_inferior,limite_superior,dimensoes,probalidade_de_abandono)

#cuckoo_min(sphere,50,10,-100,100,30,0.2)

cuckoo_min(confiabilidade,5000,100,-540,540,161,0.25)

#df_resultados1 = iterar_funcao(5,cuckoo_min,confiabilidade,2000,20,-100,100,161,0.2)
#df_resultados1.to_excel("resultados_conf200020.xlsx", index=False)
