import random
import statistics
import copy
from tqdm import tqdm
from .base_metaheuristic import BaseMetaheuristic

'''
Category:
Identifier(Ctrl+F)      Content
--------------------------------------------------------
SS              code bloques reserved for the SMA,neiborhood params
TDL              TO DO LIST
LOG              log of the modification of the main structure
'''

'''
LOG
---------------------------------------------------
1.      Cancel the optimize part
2.      The __chromosome_crossover should be implemented in the outside, but i don't know where for now
3.      Realize the solution_checker_GA in the cost() like cost_ga() or an override of cost(), or an override of the solution_check()
'''


class GeneticAlgorithm(BaseMetaheuristic):
    def __init__(self,num_evolu_per_search=10, num_parent=4, num_population=20, rate_mutation=0.2,
                 population=[], progress_bar=False, threshold=10, reproductive_isolation=False, best_seed=True,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

        self.num_evolu_per_search= num_evolu_per_search
        self.population = population
        self.rate_mutation = rate_mutation
        self.num_parent = num_parent
        self.num_population = num_population
        self.best_solution=None
        self.evolution_explored_solutions = []
        self.penalty_wrong_chromosome=40000
        self.progress_bar = progress_bar
        self.hasinit=False
        self.list_nation=[]
        self.threshold=threshold
        self.reproductive_isolation= reproductive_isolation
        self.best_seed=best_seed

    def __chromosome_mutation(self, chromosome, prob):
        if random.random() < prob and chromosome.cost() > 0:
            _, N, _ = self.get_problem_components()
            return N(chromosome)
        else:
            return chromosome

    def __chromosome_crossover(self, parent1, parent2): 

        N=self.NEIGHBORHOOD()
        N.set_params({'choose_mode':'crossover'})

        SP=self.SOLUTION_SPACE()
        if SP.distance(parent1, parent2)<self.threshold and self.reproductive_isolation==True:
            return N.get_neighbor_from_two(parent1, parent2) 
        else: 
            return self.__generate_chromosome(),self.__generate_chromosome()
            # return parent1,parent2

    def search(self):
        """ Performs metaheuristic search """
        if not self.hasinit:
            self.__init()
        iterator = tqdm(range(self.num_evolu_per_search)) if self.progress_bar else range(self.num_evolu_per_search)
        for _ in iterator:
            self.__evolution()
            
        self.evolution_best_solution.append(-self.__fitness(self.best_solution))
        self.evolution_explored_solutions = self.evolution_explored_solutions[:len(self.evolution_best_solution)]
        return self.best_solution

    def __fitness(self, solution):
        
        return -solution.cost()

    def __generate_chromosome(self):

        _, N, _ = self.get_problem_components()
        chromosome = N.initial_solution()

        return chromosome

    def __init(self):
        self.hasinit=True
        while(len(self.population)<self.num_population):
            
            new_born=self.__generate_chromosome()
            if new_born.cost()!=0:
                self.population.append(new_born)

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
            for i in range(len(parents) // 2 - 1):
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
            for i in population_new:
                self.population.append(self.__chromosome_mutation(i, self.rate_mutation))

        def __eliminate(self): 
            list_fitness = []
            for chromosome in self.population:
                fitness=self.__fitness(chromosome)
                list_fitness.append(fitness)
                if fitness >-self.penalty_wrong_chromosome:
                    self.evolution_explored_solutions.append(-fitness)
            critere = statistics.median(list_fitness)
            best_performance = max(list_fitness)

            for chromosome in self.population:
                if self.__fitness(chromosome) == best_performance:
                    self.best_solution = chromosome

            while (len(self.population) > self.num_population):
                for chromosome in self.population:
                    if self.__fitness(chromosome) <= critere:
                        self.population.remove(chromosome)

        def __regeneration(self):

            if len(self.population) < self.num_population:
                while (len(self.population) < self.num_population):
                    self.population.append(self.__generate_chromosome())

            else:
                list_fitness = []
                for chromosome in self.population:
                    list_fitness.append(self.__fitness(chromosome))

                critere = sorted(list_fitness, reverse=True)[self.num_population - 1]

                for chromosome in self.population:
                    if self.__fitness(chromosome) <= critere:
                        self.population.remove(chromosome)

                for _ in range(self.num_population - len(self.population)):
                    self.population.append(self.__generate_chromosome())

        parents = __binary_tournement(self)
        __pop_crossover(self, parents)
        __pop_mutation(self)
        __eliminate(self)
        __regeneration(self)
