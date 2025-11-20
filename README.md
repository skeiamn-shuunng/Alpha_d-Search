# Alpha_d-Search
This is a repository containing all results and code from the paper "A Note on Large Degenerate Induced Subgraphs in Sparse Graphs"

## General Information
The code and results in this repository are not proof of anything new. Instead, all information here is just evidence we have towards our conjecture. Note that an exhaustive search of every subgraph in a d-degenerate graph is an intractible problem. Thus, we used a genetic algorithm to hopefully achieve better results without having to check every possible case.

## Alpha_d_Code
In the folder "Alpha_d_Code", you will find 3 files:
- Alpha_1.py
- Alpha_d.py
- sample_batch.sh

### Alpha_1.py
Alpha_1.py is a program that finds the largest induced forest in multiple graphs and returns the smallest ordered forest it found. In order to use this program, please uncomment the last line in the program and rewrite the following parameters:
- "population_size" is the size of the initial population for your genetic algorithm
- "population_graph_order" is the number of vertices you want your starting graphs to have
- "population_graph_degeneracy" is the degeneracy of your starting graph
- "tournament_size" is how big you want the tournaments to be when picking parents
- "generation_cap" is the maximum number of generations you would like the algorithm to go through

And for more information on the actual functions in the program:
- generate_random_dgraph(n,d)
  - inputs are integers n and d
  - this function creates a d-degenerate graph of order n
- forest_fitness(G)
  - input is graph G
  - using an integer program, this function calculates the order of the largest induced forest in graph G
- fitness_function(g)
  - input is a graph6 string g
  - this function just finds the number of vertices in g not contained in the largest induced forest found in the forest_fitness function
- initialize_population(population_size,n,d)
  - inputs are integers population_size, n, and d
  - this function takes population_size and creates that many d-degenerate graphs of order n
- tournament_selection(population,fitnesses,t)
  - inputs are lists population and fitnesses, also integer t
  - this function takes the population and the fitnesses of every graph and uses those along with t in order to find some number of "parents" among the best of the generation
- crossover_ddegenerate(g,h,d)
  - inputs are graph6 strings g and h, also integer d
  - given two graphs g and h, this function creates a new d-degenerate "offspring" that contains some of the edges of g and some of the edges of h
- mutate_backedge_swap_degenerate(g,d,mutation_rate=0.05)
  - inputs are graph g, integer d, and preset mutation_rate of 0.05
  - this function takes a d-degenerate graph and randomly swaps some number of edges
- replace_bottom_k(population,fitnesses,offspring)
  - inputs are lists population, fitnesses, and offspring
  - this function removes the graphs with the lowest fitness values and replaces them with "offspring"

### Alpha_d.py
Alpha_d.py is a program that finds the largest d-degenerate induced subgraph in multiple D-degenerate graphs and returns the minimum value it found. In order to use this program, please uncomment the last line in the program and rewrite the following parameters:
- "population_size" is the size of the initial population for your genetic algorithm
- "population_graph_order" is the number of vertices you want your starting graphs to have
- "population_graph_degeneracy" is the degeneracy of your starting graph
- "alpha_d_degeneracy" is the degeneracy of the induced subgraphs that you are trying to find
- "tournament_size" is how big you want the tournaments to be when picking parents
- "generation_cap" is the maximum number of generations you would like the algorithm to go through

And for more information on the actual functions in the program:
- induced_subgraphs_n_minus_k(G,k)
  - inputs are graph G and integer k
  - this function exhaustively lists every possible subgraph of order k in graph G
- graph_degeneracy(G)
  - input is graph G
  - this function finds the degeneracy of graph G by checking for a greedy labeling of the vertices
- degen_fitness(g,d,lower,upper)
  - input graph6 string g, also integers d, lower, and upper
  - let n be the order of g. this function goes through every possible induced subgraph of g with order between n-upper and n-lower and finds the largest d-degenerate induced subgraph
- generate_random_dgraph(n,d)
  - inputs are integers n and d
  - this function creates a d-degenerate graph of order n
- fitness_function(g)
  - input is a graph6 string g
  - this function just finds the number of vertices in g not contained in the largest induced forest found in the forest_fitness function
- initialize_population(population_size,n,d)
  - inputs are integers population_size, n, and d
  - this function takes population_size and creates that many d-degenerate graphs of order n
- tournament_selection(population,fitnesses,t)
  - inputs are lists population and fitnesses, also integer t
  - this function takes the population and the fitnesses of every graph and uses those along with t in order to find some number of "parents" among the best of the generation
- crossover_ddegenerate(g,h,d)
  - inputs are graph6 strings g and h, also integer d
  - given two graphs g and h, this function creates a new d-degenerate "offspring" that contains some of the edges of g and some of the edges of h
- mutate_backedge_swap_degenerate(g,d,mutation_rate=0.05)
  - inputs are graph g, integer d, and preset mutation_rate of 0.05
  - this function takes a d-degenerate graph and randomly swaps some number of edges
- replace_bottom_k(population,fitnesses,offspring)
  - inputs are lists population, fitnesses, and offspring
  - this function removes the graphs with the lowest fitness values and replaces them with "offspring"
 
### sample_batch.sh
When running these two programs, we used the FIR Supercluster. In order to pass it any code, it must be sent to the job scheduler through a batching script. FIR uses SLURM as its job scheduler, so if one were to try and replicate our results, this is a sample batching script of the one we used to pass our jobs to SLURM.

## Run Results
To understand more about the limitations and applications of the code, please look at the Run Results folder. In here, we currently have 5 documents. 2 of them are focused on the limitations and runtime of the algorithms and the other 3 are examples of possible outputs you might receive.

### stress_testing.txt
This text file shows the results of running the faster Alpha_1.py algorithm for 100 generations with the following parameters:
- population size: 100
- order of every graph in the population: 100
- degeneracy of every graph in the population: 2
- size of tournament: 2
- generations: 100
This code took 5 hours to run. If you would like to scale up the operation, please feel free to use this estimate to do so

### stress_testing_Nonforest.txt
This text file shows the results of running the slower Alpha_d.py algorithm for 100 generations with the following parameters:
- population size: 30
- order of every graph in the population: 30
- degeneracy of every graph in the population: 3
- degeneracy of the subgraphs we are finding: 2
- size of tournament: 2
- generations: 100

This code took 8 hours to run. If you would like to scale up the operation, please feel free to use this estimate to do so

### Fifteen_Vertices_1.py and Fifteen_Vertices_3.py
These two text files show the result of running alpha_1 with the following parameters:
- population size: 100
- order of every graph in the population: 15
- degeneracy of every graph in the population: 2
- size of tournament: 2
- generations: the cap was 200, but both of these iterations found a graph with the bound we wanted at generation 6

Please note that this shows our genetic algorithm is able to achieve our bound and replicate this result.

### Fifteen_Vertices_2
This text file shows the result of running alpha_1 with the following parameters:
- population size: 100
- order of every graph in the population: 15
- degeneracy of every graph in the population: 2
- size of tournament: 2
- generations: 200

Please note that this shows our genetic algorithm is also not perfect. The random sample of graphs in the population also matter a good amount. Thus, in the actual genetic algorithm, we have included a bit of code that can semi-reset a population if it did not reach a specific bound by a certain generation.
