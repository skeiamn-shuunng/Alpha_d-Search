import networkx as nx
import copy
import random
import itertools as it
import pulp
import sys
import concurrent.futures


def generate_random_dgraph(n, d):
    """
    Input: integers n and d
    Output: a random n vertex, d-degenerate graph (as a networkx object)
    """
    assert n >= d, "n must be at least d"

    # Start with a complete graph on d nodes (labeled 0, 1, ..., d-1)
    G = nx.complete_graph(d)

    for _ in range(n - d):
        existing_vertices = list(G.nodes())
        neighbors = random.sample(existing_vertices, d)
        new_vertex = max(G.nodes()) + 1  # ensures new unique label
        G.add_node(new_vertex)
        for v in neighbors:
            G.add_edge(new_vertex, v)

        
    string_g = nx.to_graph6_bytes(G, header=False).decode('utf-8').strip()

    return string_g



def forest_fitness(G):
    """
    Input: Graph G
    Output: The order of the largest induced forest in G
    """

    # Collect all vertices in the graph
    H = G.to_directed()
    existing_vertices = sorted(set(H.nodes()))
    existing_edges = sorted(set(G.edges()))
    new_vertex = max(H.nodes())+1 #ensures new unique label
    H.add_node(new_vertex)
    vertices = sorted(set(H.nodes()))
    for vert in range(len(vertices)-1):
        H.add_edge(new_vertex,vert)
    edges = sorted(set(H.edges()))
    K_n = nx.complete_graph(len(H.nodes()), nx.DiGraph())
    all_edges = sorted(set(K_n.edges()))

    # Define the LP problem: maximize total number of selected vertices
    prob = pulp.LpProblem("TreeOrder", pulp.LpMaximize)

    # Binary decision variables: y[v] = 1 if vertex v is in the largest induced forest, else 0
    y = pulp.LpVariable.dicts("y", vertices, cat=pulp.LpBinary)

    # Binary decision variable: x[u][v] = 1 if edge uv is in the largest induced forest, else 0
    x = pulp.LpVariable.dicts("x", all_edges, cat=pulp.LpBinary)

    # Continuous variable: f[u][v]
    f = pulp.LpVariable.dicts("f", all_edges, lowBound=0, cat=pulp.LpContinuous)

    # Objective: maximize the total number of vertices selected
    prob += pulp.lpSum(y[v] for v in existing_vertices)

    # Constraint 1: sum of outneighbors of v is equal to y[v]
    for v in existing_vertices:
            prob += pulp.lpSum(x[(u,v)] for u in vertices if (u,v) in edges) == y[v]
    
    # Constraint 2: sum of the flow of out-neighbors of s is equal to sum of y[v]
    prob += pulp.lpSum(f[(new_vertex,v)] for v in existing_vertices) == pulp.lpSum(y[v] for v in existing_vertices)

    # Constraint 3: difference between flow of in and out-neighbors of v equals y[v]
    for v in existing_vertices:
        prob += pulp.lpSum(f[(u,v)] for u in vertices if (u,v) in edges) - pulp.lpSum(f[(v,w)] for w in vertices if (v,w) in edges) == y[v]
    
    # Constraint 4: y[s] = 1
    prob += y[new_vertex] == 1
    
    # Constraint 5: bounding flow
    for (u,v) in edges:
         prob += x[(u,v)] <= f[(u,v)] 
         prob += f[(u,v)] <= (len(vertices)-1)*x[(u,v)]
    
    # Constraint 6: bound on the arcs
    for (u,v) in edges:
            prob += x[(u,v)] + x[(v,u)] <= y[v]

    #Constraint 7: bound on the arcs
    for (u,v) in existing_edges:
            prob += x[(u,v)]+x[(v,u)] >= y[u] + y[v] - 1 

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Return the optimal value (size of largest induced forest)
    return int(pulp.value(prob.objective))

"""
Basic functions above,
genetic algorithm functions below.
"""

def fitness_function(g):
    """
    Computes the fitness of the graph G we are looking at. The parameter k determines which degeneracy we are assesing.
    m and M are the thresholds for the order of k-degenerate subgraphs we are looking for, and the clique_fitness we are
    assesing for respectivly.
    """   

    G = nx.from_graph6_bytes(g.encode('utf-8'))
    n = G.number_of_nodes()
    return (n - forest_fitness(G))


def initialize_population(population_size, n, d):
    """
    Creates the starting population.
    
    Inputs:
        population_size (int): How big you want the population to be
        n (int): Order of graphs in population
        d (int): The degeneracy you want
    """

    population = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(generate_random_dgraph,n,d).result() for _ in range (population_size)]
    
    for result in results:
        population.append(result)

    return population


def tournament_selection(population, fitnesses, t):
    """
    Perform tournament selection on the population.

    Inputs:
        population (list): List of individuals (G6 strings).
        fitnesses (dict): Dictionary where keys are G6 strings (individuals),
                          and values are fitness scores.
        t (int): Tournament size.

    Returns:
        list: Selected individuals (parents).
    """
    selected = []
    adjoin = list(zip(population,fitnesses))
    parent_amt = len(adjoin)//t
    
    for _ in range(parent_amt):  # Select two parents
        # Randomly pick t individuals from the population
        candidates = random.sample(adjoin, t)
        # Select the one with the highest fitness
        best = max(candidates, key=lambda ind: ind[1])
        selected.append(best[0])
        adjoin.remove(best)
    
    return selected


def crossover_ddegenerate(g, h, d):
    """
    Crossover two d-degenerate graphs G and H with same vertex ordering (0..n-1),
    and return a child graph Q that is also d-degenerate.

    g (G) and h (H) are inputed as graph6 strings
    
    The crossover is based on combining backward neighbors of each vertex using:
    - Intersection of neighbors in G and H
    - Random choice from the symmetric difference until d neighbors are selected
    """

    G = nx.from_graph6_bytes(g.encode('utf-8'))
    H = nx.from_graph6_bytes(h.encode('utf-8'))
    
    n = G.number_of_nodes()
    Q = nx.Graph()
    Q.add_nodes_from(range(n))

    for v in range(n):
        preds = set(range(v))  # backward vertices

        # Get backward neighbors in G and H
        ng = set(G.neighbors(v)) & preds
        nh = set(H.neighbors(v)) & preds

        shared = ng & nh
        symdiff = ng.symmetric_difference(nh)

        # Start with shared neighbors
        neighbors = set(shared)

        # Fill up to d with random from symmetric difference
        remaining = len(ng) - len(neighbors)
        if remaining > 0:
            candidates = list(symdiff)
            if len(candidates) < remaining:
                raise ValueError(f"Cannot assign {d} neighbors to vertex {v}; insufficient distinct candidates.")
            neighbors.update(random.sample(candidates, remaining))

        # Add edges to Q
        for u in neighbors:
            Q.add_edge(u, v)

    q = nx.to_graph6_bytes(Q, header=False).decode('utf-8').strip()

    return q


def mutate_backedge_swap_ddegenerate(g, d, mutation_rate=0.05):
    """
    For each vertex v, with probability mutation_rate:
    - remove k backward edges (k selected randomly),
    - add k backward edges,
    so that the number of backward neighbors of v remains the same.

    g (G) is inputed as graph6 strings
    """
    H = nx.from_graph6_bytes(g.encode('utf-8'))

    n = H.order()

    for v in range(n):
        if random.random() < mutation_rate:
            preds = set(range(v))
            backward_neighbors = set(H.neighbors(v)) & preds
            non_neighbors = preds - backward_neighbors

            max_k = min(len(backward_neighbors), len(non_neighbors))
            if max_k == 0:
                continue  # Nothing to swap

            k = random.randint(1, max_k)

            # Choose edges to remove and add
            to_delete = random.sample(list(backward_neighbors), k)
            to_add = random.sample(list(non_neighbors), k)

            # Perform swap
            for u in to_delete:
                H.remove_edge(u, v)
            for u in to_add:
                H.add_edge(u, v)

    h = nx.to_graph6_bytes(H, header=False).decode('utf-8').strip()
    return h


def replace_bottom_k(population, fitnesses, offspring):
    """
    Replace the bottom k individuals in the population with offspring.

    Inputs:
        population (list): Current population.
        fitnesses (dict): Dictionary where keys are G6 strings (individuals),
                          and values are fitness scores.
        offspring (list): New individuals to insert.

    Returns:
        tuple: (new_population, new_fitnesses)
    """
    k = len(offspring)
    if k > len(population):
        raise ValueError("Offspring size larger than population size.")

    # Pair population with fitness
    paired = list(zip(population, fitnesses))

    # Sort by fitness ascending (worst first)
    paired.sort(key=lambda x: x[1])

    # Remove bottom k
    paired = paired[k:]

    # Add offspring with dummy fitnesses (e.g., None or you can compute fitness separately)
    # Here we set offspring fitnesses to None, so user should update fitnesses later
    offspring_with_fitness = [(ind, None, None) for ind in offspring]
    paired.extend(offspring_with_fitness)

    # Unzip back to separate lists
    new_population, new_fitnesses = zip(*paired)
    
    return list(new_population), list(new_fitnesses)

def fitness_finder(population,fitness):
    #Find the fitness of a graph
    if fitness == None:
        fitness = fitness_function(population)
    return fitness


def genetic_algorithm(population_size,n,d,tournament_size, gen_cap):
    """
    The main loop. Generates graphs, checks fitness, iterates until the goal is reached or the generation reaches the gen_cap

    Inputs:
        population_size (int): How big you want the population to be
        n (int): Order of the graphs in the population
        d (int): The degeneracy of your starting graphs
        tournament_size (int): How many graphs you want to compare to be parents
        gen_cap: the integer denoting the stopping point
    """
    #Initialize population
    population = initialize_population(population_size,n,d)

    #Set up the champion
    champion = (None, 0.0)

    #Generation and the infinite condition
    gen = 0
    fitnesses = [None]*population_size

    #The start of the loop
    while (gen < gen_cap):
        #If any fitness is unavailable, calculate it
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(fitness_finder,population,fitnesses)

            fitnesses.clear()
            
            for result in results:
                fitnesses.append(result)

        #Culling function
        if (gen == 100):
            if (champion[1]/n >= 7/10):
                temp_pop = initialize_population(population_size//2,n,d)
                population, fitnesses = replace_bottom_k(population, fitnesses, temp_pop)
                gen = 0
            
        #Store the best performer of the current generation
        best_individual = max(zip(population,fitnesses), key = lambda x: x[1])

        #Start selecting the best of each generation
        selected = tournament_selection(population, fitnesses, tournament_size)
        offspring = []
        child = []
        
        #Pick best parents and create offspring
        parent1 = []
        parent2 = []
        for i in range(0, len(selected), 2):
            parent1.append(selected[i])
            parent2.append(selected[i+1])
        with concurrent.futures.ProcessPoolExecutor() as executor:
            part1 = executor.map(crossover_ddegenerate,parent1,parent2,it.repeat(d))

        for result in part1:
            child.append(result)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            part2 = executor.map(mutate_backedge_swap_ddegenerate,child,it.repeat(d))

        for result in part2:
            offspring.append(result)

        #Replace population and print best of generation
        population, fitnesses = replace_bottom_k(population, fitnesses, offspring)

        #Check current bound
        if best_individual[1] > champion[1]:
            champion = best_individual
            print(gen,champion[0],champion[1])
        gen+=1
        
        if gen%50 == 0:
            print(f"This generation is: {gen}")
        #Termination condition
        if ((n-best_individual[1])/n <= 3/5):
            print(f"This graph is better than our bound: {best_individual[0]}")
            break

    return population

#print(genetic_algorithm(population_size,population_graph_order,population_graph_degeneracy,tournament_size,generation_cap)) #Uncomment and replace parameters in this line when using the code
