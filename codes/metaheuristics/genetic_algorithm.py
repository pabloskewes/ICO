import random
import copy
from tqdm import tqdm
from .base_metaheuristic import BaseMetaheuristic
import matplotlib.pyplot as plt

class GeneticAlgorithm(BaseMetaheuristic):
    def __init__(self,num_evolu_per_search=10, num_parent=4, num_population=20, rate_mutation=0.2,rate_crossover=0.7,
                 population=[], progress_bar=False, threshold=10, reproductive_isolation=False, best_seed=True,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

        self.num_evolu_per_search= num_evolu_per_search
        self.population = population
        self.rate_mutation = rate_mutation
        self.rate_crossover= rate_crossover
        self.num_parent = num_parent
        self.num_population = num_population
        self.best_solution=None
        self.evolution_explored_solutions = []
        self.progress_bar = progress_bar
        self.has_init=False
        self.list_nation=[]
        self.threshold=threshold
        self.reproductive_isolation= reproductive_isolation
        self.best_seed=best_seed
        self.avr_cost=0
        self.evolution_avr_solutions=[]
        self.evolution_best_solutions=[]

    def plot_evolution_cost(self, figsize=(20, 10)):
        # plt.scatter(x=list(range(len(self.cost_list_best_sol))), y=self.cost_list_best_sol, c='turquoise')
        plt.figure(figsize=figsize)
        plt.title('Evolution of the cost of the found solutions')
        plt.plot(self.evolution_explored_solutions, c='turquoise', label='explored solutions')
        plt.plot(self.evolution_best_solutions, c='orange', label='best solution')
        plt.plot(self.evolution_avr_solutions, c='blue', label='avr solution')
        plt.xlabel('Time (iteration)')
        plt.ylabel('Cost of the solution')
        plt.legend()
        plt.show()

    def search(self):
        """ Performs metaheuristic search """
        if not self.has_init:
            self.__init()

        pbar = tqdm(total=self.num_evolu_per_search)if self.progress_bar else None

        if self.progress_bar and self.best_solution:
            pbar.set_description('Cost: %.2f' %self.best_solution.cost())

        for _ in range(self.num_evolu_per_search):
            self.__evolution()
            self.evolution_avr_solutions.append(self.avr_cost)
            self.evolution_best_solutions.append(-self.__fitness(self.best_solution))
            self.evolution_explored_solutions = self.evolution_explored_solutions[:len(self.evolution_best_solutions)]

            if self.progress_bar:
                pbar.update()
                pbar.set_description('Cost: %.2f' % self.best_solution.cost())

        if self.progress_bar:
            pbar.close()

        return copy.deepcopy(self.best_solution)

    def __fitness(self, solution):
        
        return -solution.cost()

    def __generate_chromosome(self):

        _, N, _ = self.get_problem_components()
        chromosome = N.initial_solution()
        
        return chromosome

    def __chromosome_mutation(self, chromosome):
        if random.random() < self.rate_mutation and self.__fitness(chromosome) != 0:
            _, N, _ = self.get_problem_components()
            return N(chromosome)
        else:
            return chromosome

    def __chromosome_crossover(self, parent1, parent2): 
        if random.random() < self.rate_crossover:
            N=self.NEIGHBORHOOD()
            N.set_params({'choose_mode':'crossover'})

            SP=self.SOLUTION_SPACE()
            if SP.distance(parent1, parent2)<self.threshold and self.reproductive_isolation==True:
                return N.get_neighbor_from_two(parent1, parent2) 
            else: 
                return self.__generate_chromosome(),self.__generate_chromosome()
        else:
            return parent1,parent2

    def __init(self):
        self.has_init=True

        while(len(self.population)<self.num_population):
            
            new_born=self.__generate_chromosome()
            if self.__fitness(new_born)!=0:
                self.population.append(new_born)
                self.best_solution=new_born

    def __evolution(self): 

        def __binary_tournement(self):
            parents = []
            for i in range(self.num_parent):
                candidate = random.sample(self.population, 2)
                if self.__fitness(candidate[0]) > self.__fitness(candidate[1]):
                    if random.random() < 0.95:
                        parents.append(candidate[0])
                    else:
                        parents.append(candidate[1])
                else:
                    if random.random() < 0.95:
                        parents.append(candidate[1])
                    else:
                        parents.append(candidate[0])

            return parents

        def __pop_crossover(self, parents): 
            for _ in range(len(parents) // 2 - 1):
                parent = random.sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], parent[1])

                self.population.append(child1)
                self.population.append(child2)

            if self.best_solution and self.best_seed:
                parent = random.sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], self.best_solution)

            else:
                parent = random.sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], parent[1])

            self.population.append(child1)
            self.population.append(child2)

        def __pop_mutation(self):  
            population_new = copy.deepcopy(self.population)
            self.population = []
            for chromosome in population_new:
                self.population.append(self.__chromosome_mutation(chromosome))

        def __eliminate(self): 
            list_fitness = []
            for chromosome in self.population:
                fitness=self.__fitness(chromosome)
                list_fitness.append(fitness)
            
            list_fitness.sort()
            if self.num_population<len(list_fitness):
                critere = list_fitness[-self.num_population//2]
            best_performance = max(list_fitness)

            for chromosome in self.population:
                if self.__fitness(chromosome)<= critere:
                    self.population.remove(chromosome)
                if self.__fitness(chromosome) == best_performance:
                    self.best_solution = chromosome

        def __regeneration(self):

            if len(self.population) <= self.num_population:
                while (len(self.population) < self.num_population):
                    self.population.append(self.__generate_chromosome())
            for chromosome in self.population:
                self.avr_cost+=chromosome.cost()
            self.avr_cost/=self.num_population

        parents = __binary_tournement(self)
        __pop_crossover(self, parents)
        __pop_mutation(self)
        __eliminate(self)
        __regeneration(self)
