import random
import numpy as np 
from math import pi, sin, cos 
from scipy.special import gamma 
from matplotlib import pyplot as plt 
import time  # Importa o módulo time
import copy
import pandas as pd  # Certifique-se de importar o pandas no início do arquivo

class Graph_controller(object):
    def __init__(self):
        pass

    @staticmethod
    def plot_sch(nests):
        """
        Plota todos os ninhos para a função Schaffer, limitando o intervalo até x = 1.
        """
        # Extrai os valores de fitness de todos os ninhos
        fitness1 = [nest.fitness[0] for nest in nests]
        fitness2 = [nest.fitness[1] for nest in nests]

        # --- Adicionando a verdadeira fronteira de Pareto da Schaffer min-min ---
        x_true = np.linspace(0, 2, 500)  # Intervalo ajustado para [0, 1]
        f1_true = x_true ** 2
        f2_true = (x_true - 2) ** 2
        plt.plot(f1_true, f2_true, 'r-', linewidth=2, label="Fronteira de Pareto Verdadeira (Schaffer)", zorder=1)

        # --- Plotando os pontos calculados ---
        plt.scatter(fitness1, fitness2, s=15, color="blue", label="Todos os Ninhos", zorder=2)

        # --- Configurações do gráfico ---
        plt.xlabel("f1(x) = x²")
        plt.ylabel("f2(x) = (x-2)²")
        plt.title("Fronteira de Pareto - Comparação com Schaffer min-min")
        plt.legend()
        plt.grid(True)  # Adiciona grade para melhor visualização
        plt.savefig("pareto_plot_schaffer_all.png")
        plt.show()

    @staticmethod
    def plot_zdt1(nests):
        """
        Plota todos os ninhos para a função ZDT1.
        """
        # Extrai os valores de fitness de todos os ninhos
        fitness1 = [nest.fitness[0] for nest in nests]
        fitness2 = [nest.fitness[1] for nest in nests]

        # --- Adicionando a fronteira de Pareto verdadeira da ZDT1 ---
        x_true = np.linspace(0, 1, 500)  # Intervalo onde está a fronteira de Pareto
        f1_true = x_true
        f2_true = 1 - np.sqrt(x_true)
        plt.plot(f1_true, f2_true, 'r-', linewidth=2, label="Fronteira de Pareto Verdadeira (ZDT1)", zorder=1)

        # --- Plotando os pontos calculados ---
        plt.scatter(fitness1, fitness2, s=10, color="blue", label="Todos os Ninhos", zorder=2)

        # --- Configurações do gráfico ---
        plt.xlabel("f1(x)")
        plt.ylabel("f2(x)")
        plt.title("Fronteira de Pareto - Comparação com ZDT1")
        plt.legend()
        plt.grid(True)  # Adiciona grade para melhor visualização
        plt.savefig("pareto_plot_zdt1_all.png")
        plt.show()
        
    @staticmethod

    def plot_zdt3(nests):
        """
        Plota todos os ninhos para a função ZDT3.
        """

        # Extrai os valores de fitness de todos os ninhos
        fitness1 = [nest.fitness[0] for nest in nests]
        fitness2 = [nest.fitness[1] for nest in nests]

        # --- Gera a verdadeira fronteira de Pareto da ZDT3 ---
        x_true = np.linspace(0, 1, 1000)
        f1_true = x_true
        f2_true = 1 - np.sqrt(x_true) - x_true * np.sin(10 * np.pi * x_true)

        # Divide a frente de Pareto em segmentos para lidar com discontinuidade
        segments = []
        current_segment = []
        for i in range(len(f1_true) - 1):
            current_segment.append((f1_true[i], f2_true[i]))
            # Verifica se há uma discontinuidade (grande salto)
            if abs(f2_true[i + 1] - f2_true[i]) > 0.2:  # Aumentado o limite para capturar os saltos reais
                segments.append(current_segment)
                current_segment = []
        current_segment.append((f1_true[-1], f2_true[-1]))
        segments.append(current_segment)

        # Plota cada segmento da frente de Pareto
        for segment in segments:
            segment = np.array(segment)
            plt.plot(segment[:, 0], segment[:, 1], 'r-', linewidth=2, label="Fronteira de Pareto Verdadeira (ZDT3)")

        # --- Plotando os pontos dos ninhos ---
        plt.scatter(fitness1, fitness2, s=10, color="blue", label="Todos os Ninhos", zorder=2)

        # --- Configurações do gráfico ---
        plt.xlabel("f1(x)")
        plt.ylabel("f2(x)")
        plt.title("Fronteira de Pareto - Comparação com ZDT3")
        plt.legend()
        plt.grid(True)
        plt.savefig("pareto_plot_zdt3_all.png")
        plt.show()

class Fitness_controller(object):
    @staticmethod
    def zdt1(vector, dimension):
        f1 = vector[0]
        g = 1.0 + ((9.0 * np.sum(vector[1:])) / (dimension - 1))
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([f1, f2])
    
    @staticmethod
    def sch(vector,dimension):
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
        """
        ZDT3 function (discontinuous Pareto front).
        """
        f1 = vector[0]
        g = 1.0 + 9.0 * np.sum(vector[1:]) / (dimension - 1)
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)  # Termo sinusoidal adicional
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

class Environment:
    """
    Classe para encapsular as variáveis globais compartilhadas entre as classes.
    """
    def __init__(self, dimension, lower_bound, upper_bound, fitness_function):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fitness_function = fitness_function
        

class Nest:
    def __init__(self, env: Environment):
        self.env = env
        self.rank = None
        self.distance = 0.0
        self.function_parameters = self.create_function_parameters()
        self.fitness = self.calculate_fitness()

    def create_function_parameters(self):
        return np.round(np.random.uniform(self.env.lower_bound, self.env.upper_bound, self.env.dimension), 5)

    def calculate_fitness(self):
        return self.env.fitness_function(self.function_parameters, self.env.dimension)

    def levy_flight(self, gbest):
        """
        Aplica o voo de Lévy para os parâmetros do ninho usando o gbest fornecido.
        """
        if gbest is None:
            return

        alpha = 0.1
        beta = 1.5
        sigma = (gamma(1 + beta) * sin(pi * beta / 2.0) /
                 (gamma((1 + beta) / 2) * beta * 2.0 ** ((beta - 1) / 2.0))) ** (1.0 / beta)

        u = np.random.normal(0., sigma, np.shape(self.function_parameters))
        v = np.random.normal(0., 1., np.shape(self.function_parameters))
        levy_step = (u / np.abs(v) ** (1 / beta)) * (self.function_parameters - gbest.function_parameters)
        self.function_parameters += alpha * levy_step

        # Verifica se os parâmetros ultrapassaram os limites
        for i in range(len(self.function_parameters)):
            if self.function_parameters[i] < self.env.lower_bound or self.function_parameters[i] > self.env.upper_bound:
                self.function_parameters[i] = np.random.uniform(self.env.lower_bound, self.env.upper_bound)

        self.fitness = self.calculate_fitness()


class Farmer:
    def __init__(self, population_size, abandon_rate, env: Environment):
        self.env = env
        self.nests = self.create_nests(population_size)
        self.abandon_rate = abandon_rate
        self.population_size = population_size
        self.gbest = None


    def create_nests(self, population_size):
        return [Nest(self.env) for _ in range(population_size)]

    def not_dominated(self, nest1, nest2):
        """
        Verifica se nest1 não é dominado por nest2.
        """
        return np.all(nest1.fitness <= nest2.fitness) and np.any(nest1.fitness < nest2.fitness)

    def rank_nests(self):
        """
        Rankeia os ninhos com base na dominância.
        """
        for nest in self.nests:
            nest.rank = None

        current_rank = 0
        remaining_nests = self.nests[:]

        while remaining_nests:
            non_dominated = []
            for nest in remaining_nests:
                is_dominated = False
                for other_nest in remaining_nests:
                    if nest != other_nest and self.not_dominated(other_nest, nest):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(nest)

            for nest in non_dominated:
                nest.rank = current_rank

            remaining_nests = [nest for nest in remaining_nests if nest not in non_dominated]
            current_rank += 1

    def select_gbest(self):
        """
        Seleciona o melhor ninho como gbest.
        O melhor ninho é o primeiro da lista geral de ninhos.
        """
        if self.nests:
            self.gbest = self.nests[0]  # Seleciona o primeiro ninho da lista
        else:
            self.gbest = None  # Caso a lista esteja vazia

    def apply_levy_flight(self):
        """
        Aplica o voo de Lévy para todos os ninhos usando o gbest selecionado.
        Cria uma cópia da população, aplica o voo de Lévy na cópia e combina com a população original.
        """
        # Cria uma cópia profunda da população atual
        new_population = [copy.deepcopy(nest) for nest in self.nests]

        # Aplica o voo de Lévy na nova população
        for nest in new_population:
            nest.levy_flight(self.gbest)

        # Combina a população original com a nova
        self.nests.extend(new_population)

    def abandon_nests(self):
        """
        Abandona os ninhos com base na taxa de abandono.
        """
        num_to_abandon = int(len(self.nests) * self.abandon_rate)
        self.nests.sort(key=lambda nest: nest.rank)
        self.nests = self.nests[:-num_to_abandon]
        self.nests.extend(self.create_nests(num_to_abandon))

    def save_pareto_to_excel(self, filename="nests_data.xlsx"):
        """
        Salva todos os ninhos em um arquivo .xlsx, incluindo rank, distance, fitness e parâmetros.
        """
        data = []
        for nest in self.nests:
            # Cria uma linha com rank, distance, fitness e parâmetros
            row = [nest.rank, nest.distance] + list(nest.fitness) + list(nest.function_parameters)
            data.append(row)

        # Define os nomes das colunas
        num_params = len(self.nests[0].function_parameters) if self.nests else 0
        columns = ["Rank", "Distance", "f1", "f2"] + [f"x{i+1}" for i in range(num_params)]

        # Cria o DataFrame e salva no Excel
        df = pd.DataFrame(data, columns=columns)
        df.to_excel(filename, index=False)
        print(f"Dados de todos os ninhos salvos em: {filename}")

    def crowding_distance(self):
        """
        Calcula as distâncias de aglomeração para os ninhos em cada rank.
        """
        # Agrupa os ninhos por rank
        max_rank = max(nest.rank for nest in self.nests if nest.rank is not None)
        for rank in range(max_rank + 1):
            # Filtra os ninhos do rank atual
            rank_nests = [nest for nest in self.nests if nest.rank == rank]

            # Inicializa a distância de aglomeração
            for nest in rank_nests:
                nest.distance = 0.0

            if len(rank_nests) == 0:
                continue

            num_objectives = len(rank_nests[0].fitness)

            # Calcula a distância de aglomeração para cada objetivo
            for i in range(num_objectives):
                # Ordena os ninhos pelo valor do fitness no objetivo i
                rank_nests.sort(key=lambda nest: nest.fitness[i])

                # Define as distâncias infinitas para os extremos
                rank_nests[0].distance = float('inf')
                rank_nests[-1].distance = float('inf')

                # Calcula a distância normalizada para os ninhos intermediários
                min_value = rank_nests[0].fitness[i]
                max_value = rank_nests[-1].fitness[i]
                range_value = max_value - min_value

                if range_value == 0:
                    continue

                for j in range(1, len(rank_nests) - 1):
                    next_fitness = rank_nests[j + 1].fitness[i]
                    prev_fitness = rank_nests[j - 1].fitness[i]
                    rank_nests[j].distance += (next_fitness - prev_fitness) / range_value

    def abandon_half(self):
        """
        Abandona metade dos ninhos (os últimos da lista).
        """
        num_to_abandon = len(self.nests) // 2  # Calcula metade dos ninhos
        self.nests = self.nests[:-num_to_abandon]  # Remove os últimos ninhos

    def abandon_by_rate(self):
        """
        Abandona os ninhos com base na probabilidade de abandono (similar ao MATLAB).
        Substitui os ninhos abandonados por novos ninhos gerados por passos aleatórios.
        """
        d = self.env.dimension

        # Cria duas listas auxiliares embaralhadas
        shuffled_nests1 = random.sample(self.nests, len(self.nests))  # Primeira cópia embaralhada
        shuffled_nests2 = random.sample(self.nests, len(self.nests))  # Segunda cópia embaralhada

        # Gera a matriz de probabilidade de abandono
        for i, nest in enumerate(self.nests):
            if random.random() < self.abandon_rate:  # Probabilidade de abandono
                # Seleciona os ninhos correspondentes das listas embaralhadas
                nest1 = shuffled_nests1[i]
                nest2 = shuffled_nests2[i]
                # Calcula o passo aleatório
                stepsize = np.random.rand(d) * (nest1.function_parameters - nest2.function_parameters)
                # Atualiza os parâmetros do ninho
                new_parameters = nest.function_parameters + stepsize
                # Garante que os parâmetros estejam dentro dos limites
                new_parameters = np.clip(new_parameters, self.env.lower_bound, self.env.upper_bound)
                # Atualiza o ninho com os novos parâmetros
                nest.function_parameters = new_parameters
                nest.fitness = nest.calculate_fitness()

    def sort_nests(self):
        """
        Organiza os ninhos primeiro pelo menor rank e, dentro de cada rank, pela maior distância.
        """
        self.nests.sort(key=lambda nest: (nest.rank, -nest.distance))

def main(max_generation, population_size, abandon_rate, dimension, lower_bound, upper_bound, fitness_function):
    """
    Função principal para executar o algoritmo.
    """
    start_time = time.time()

    env = Environment(dimension, lower_bound, upper_bound, fitness_function)
    alegrio = Farmer(population_size, abandon_rate, env)

    for i in range(max_generation):
        alegrio.select_gbest()
        alegrio.apply_levy_flight()
        alegrio.abandon_by_rate()
        alegrio.rank_nests()
        alegrio.crowding_distance()
        alegrio.sort_nests()
        alegrio.abandon_half()
    
        print(f"Geração: {i + 1}/{max_generation} | Número de Ninhos: {len(alegrio.nests)} | Melhor Ninho: {alegrio.gbest.fitness}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo total de execução: {elapsed_time:.2f} segundos")

    # Salva e plota todos os ninhos
    if len(alegrio.nests) > 0:
        alegrio.save_pareto_to_excel()  # Salva todos os ninhos no Excel
        Graph_controller.plot_sch(alegrio.nests)  # Plota todos os ninhos
    else:
        print("Nenhum ninho disponível para salvar ou plotar.")
# Configuração e execução do algoritmo
main(
    max_generation=5000,
    population_size=100,
    abandon_rate=0.1,
    dimension=1,
    lower_bound=-1000,
    upper_bound=1000,
    fitness_function=Fitness_controller.sch
)