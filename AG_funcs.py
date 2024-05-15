import networkx as nx
import random
from glob import glob
import json
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

VARIABLE_MAP = {0: 'Preco_L',
                1: 'SIN-EA',
                2: 'SIN-DM',
                3: 'TER',
                4: 'NUC',
                5: 'HID',
                6: 'EOL',
}

def get_key(value, v_map):
    key_list = list(v_map.keys())
    val_list = list(v_map.values())
    position = val_list.index(value)

    return(key_list[position])

def names_to_map(topology, v_map):
    output = []
    for edge in topology:
        output.append([get_key(edge[0], v_map), get_key(edge[1], v_map)])

    return output

def get_available_names(pop_size):
    universe_set = set([i for i in range(1, pop_size + 1)])

    no_available_set = set([int(element.replace('ind/ind_','').replace('.json', '')) for element in glob('ind/*.json')])

    available_set = universe_set - no_available_set

    return tuple(f'ind_{number:03d}.json' for number in available_set)

def get_random_topology_edges(n_edges, n_nodes, variable_map):

  edges = []
  while 0 not in np.array(list(edges)).flatten():
    random_selected_nodes = list(random.sample(list(variable_map.keys()), k=n_nodes))
    edges = set()

    while len(edges) < n_edges:
        aux_edge = tuple(sorted(random.sample(random_selected_nodes, k=2)))
        edges.add(aux_edge)

  return list(edges)

# Função para criar uma população inicial de indivíduos
def create_initial_graph_population(pop_size, n_edges, n_nodes, variable_map):
    population = []
    for _ in range(pop_size):
        edges = get_random_topology_edges(n_edges, n_nodes, variable_map)
        G = nx.DiGraph(edges)
        population.append(G)
    return population

def write_individuals(graph_population, pop_size):
    for individual in graph_population:
        topology = []
        for edge in individual.edges:
            topology.append([VARIABLE_MAP[edge[0]], VARIABLE_MAP[edge[1]]])
        
        try:
            name = get_available_names(pop_size)[0]
        except IndexError:
            print('Warning: Trying to write more individuals than available names for this population size.')
            return None
        
        ind = {
            "name": name.split('.')[0],
            "topologia": topology,
            "n_cat": 5,#alterar depoiss
            "norm_type": "qt",
            "target_variable": "Preco_L",
            "n_years": 6,
            "data_source": "df_p_dcide.parquet",
            "starting_year": 2017
        }

        with open(f'ind/{name}', 'w') as f:
            json.dump(ind, f, indent=4)
        
# Função para realizar o crossover entre dois pais e gerar dois filhos (crossover simples)
def crossover_simples(parent1, parent2):
    child1 = nx.compose(parent1, parent2)
    child2 = nx.compose(parent2, parent1)
    return child1, child2

# Função para realizar o crossover entre dois pais e gerar dois filhos (crossover uniforme)
def crossover(parent1, parent2):
    # Obter listas de arestas dos pais
    edges_parent1 = list(parent1.edges)
    edges_parent2 = list(parent2.edges)

    # Tamanho da solução
    solution_size = len(edges_parent1)

    # Ponto de corte aleatório
    crossover_point = random.randint(0, solution_size)

    # Inicializar filhos com genes dos pais
    child1_edges = edges_parent1[:crossover_point] + edges_parent2[crossover_point:]
    child2_edges = edges_parent2[:crossover_point] + edges_parent1[crossover_point:]

    # Criar grafos a partir das listas de arestas
    child1 = nx.DiGraph(child1_edges)
    child2 = nx.DiGraph(child2_edges)

    return child1, child2

# Função para realizar a mutação em um indivíduo
def mutate(individual, mutation_rate, n_edges, n_nodes, variable_map):
    if random.random() < mutation_rate:
        edges = get_random_topology_edges(n_edges, n_nodes, variable_map)
        mutated_individual = nx.DiGraph(edges)
    else:
        mutated_individual = individual.copy()
    return mutated_individual

def get_population():
    paths = glob('ind/*.json')
    population = []
    for path in paths:
        ind_dict = ''
        with open(path, 'r') as f:
            ind_dict = json.load(f)

        #edges = ind_dict['topologia']

        # g = nx.DiGraph()
        # g.add_edges_from(names_to_map(edges, variable_map))
        # population.append(g)
            
        population.append(ind_dict)

    return population

def get_scores(show_complete=False):
    score_list = []
    complete_score_list = []
    for path in glob('scr/*.json'):
        with open(path) as f:
          score = json.load(f)
          score_list.append(score['score'])
          complete_score_list.append(score)

    if show_complete:
        return complete_score_list
    
    return score_list
# Função para selecionar os melhores indivíduos da população usando roleta

def select_best_individuals_roulette(num_best, delete_non_selected=True):
    
    population = get_population()
    scores = get_scores()

    evaluations = [(ind, score) for ind, score in zip(population, scores)]
    total_fitness = sum(fitness for _, fitness in evaluations)
    probabilities = [fitness / total_fitness for _, fitness in evaluations]
    best_individuals = random.choices(population, weights=probabilities, k=num_best)

    unique_best_individuals = []
    seen = set()

    for item in best_individuals:
        if item['name'] not in seen:
            unique_best_individuals.append(item)
            seen.add(item['name'])

    if delete_non_selected:
        selected_paths = [glob('ind/' + individual['name'] + '.json')[0] for individual in unique_best_individuals]
        selected_paths.extend([glob('scr/' + individual['name'].replace('ind', 'scr') + '.json')[0] for individual in unique_best_individuals])
        paths = glob('ind/*.json') + glob('scr/*.json')

        for path in paths:
            if path not in selected_paths:
                os.remove(path)
    
    best_individuals_index = [int(individual['name'].split('_')[-1]) for individual in unique_best_individuals]

    # with open('dont_calculate_again.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(best_individuals_index)
    

    return unique_best_individuals

def draw_topology(G, figsize=(12, 12)):
    plt.figure(figsize=figsize, dpi=150)
    pos = nx.spring_layout(G)  # Usando spring_layout para obter uma posição flexível
    nx.draw(
        G,
        pos,
        arrowsize=12,
        with_labels=True,
        node_size=8000,
        node_color="#ffff8f",
        linewidths=2.0,
        width=1.5,
        font_size=14,
    )
    plt.show()

def convert_dict_population_to_graph(population):
    graph_pop = []
    for individual in population:
        edges = individual['topologia']
        g = nx.DiGraph()
        g.add_edges_from(names_to_map(edges, VARIABLE_MAP))
        graph_pop.append(g)
    return graph_pop