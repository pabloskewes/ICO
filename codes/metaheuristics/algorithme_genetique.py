import random
import re
import statistics
import pandas as pd
import os
import sys
from math import sqrt
import copy
import numpy as np
from .base_metaheuristic import BaseMetaheuristic
from .base_problem import Problem

# from solution import sol_to_list_routes
'''
example:
from vrptw import VRPTW
context=VRPTW(load_solomon('simple.csv', nb_cust=10, vehicle_speed=100))
ga=GeneticAlgorithm()
ga.params=None
ga.context=context
ga.init() # initialize the population of ga
ga.search()# return the best solution ever found 
'''
'''
Category:
Identifier(Ctrl+F)      Content
--------------------------------------------------------
SS              code bloques reserved for the SMA,neiborhood params
TDL              TO DO LIST
KK               methode that should be kept here
AA               Methode that should be rewrite in the 
LOG              log of the modification of the main structure
'''
#TDL:
##sol_to_list_routes ,solution_checker_ga :__fitness and solution_check_ga
#solution.cost(), solution_checker()
# fitness() plays with cost

# ga=GeneticAlgorithm()
# ga.context=load_solomon()

'''
LOG
---------------------------------------------------
1.      Cancel the optimize part
2.      The __chromosome_crossover should be implemented in the outside, but i don't know where for now
3.      Realize the solution_checker_GA in the cost() like cost_ga() or an override of cost(), or an override of the solution_check()
'''

class GeneticAlgorithm(BaseMetaheuristic):
    def __init__(self,solution_initial, num_parent=4, num_population=20, rate_mutation=0.2,
                 population=[],solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}
        # self.best_solution=[0, 5, 3, 7, 8, 9, 6, 4, 10, 1, 2, 0]
        self.modele_chromosome = solution_initial
        self.population = population
        self.rate_mutation = rate_mutation
        self.num_parent = num_parent
        self.num_population = num_population
        self.best_solutions = {}

#KK
    def __get_best_solution(self):
        if self.best_solution:
            return self.best_solution
        else:
            print("Empty solution. To get the best solution, use search() first")
            return False

#AA
    def __chromosome_mutation(self, chromosome, prob):
        if random.random()<prob:
            return self.NEIGHBORHOOD.get_neighbor(chromosome)
        else:
            return chromosome

        if random.random() < prob:
            dice = random.random()
            # SS
            if dice < 0.5:
                head = random.randrange(1, len(chromosome))
                end = random.randrange(head, len(chromosome))
                tmp = chromosome[head:end]
                tmp.reverse()
                result = chromosome[:head] + tmp + chromosome[end:]
                return result
            elif dice >= 0.5:
                head = random.randrange(1, len(chromosome))
                end = random.randrange(head, len(chromosome))
                tmp = chromosome[head]
                chromosome[head] = chromosome[end]
                chromosome[end] = tmp
                return chromosome
        else:
            return chromosome

#AA
    def __chromosome_crossover(self, parent1, parent2): #what about cross-mute?

        def process_gen_repeated(copy_child1, copy_child2):
            count1 = 0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:  # If need to fix repeated gen
                    count2 = 0
                    for gen2 in parent1[pos:]:  # Choose next available gen
                        if gen2 not in copy_child1:
                            child1[count1] = parent1[pos:][count2]
                        count2 += 1
                count1 += 1
            count1 = 0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:  # If need to fix repeated gen
                    count2 = 0
                    for gen2 in parent2[pos:]:  # Choose next available gen
                        if gen2 not in copy_child2:
                            child2[count1] = parent2[pos:][count2]
                        count2 += 1
                count1 += 1
            return child1, child2

        pos = random.randrange(1, len(self.modele_chromosome) - 1)
        child1 = parent1[:pos] + parent2[pos:]
        child2 = parent2[:pos] + parent1[pos:]

        return process_gen_repeated(child1, child2)

    def search(self):
        """ Performs metaheuristic search """
        _, N, _ = self.get_problem_components()
        init_sol = N.initial_solution()
        self.best_solution=init_sol
        self.evolution_best_solution.append(self.__fitness(self.best_solution))

        for i in range(200):
            self.__evolution()

        self.evolution_best_solution.append(self.__fitness(self.best_solution))
        return self.best_solution

    def __fitness(self, solution):
        """
        is the fitness always has the same definition with the cost of the solution?
        """
        
        if rout(self.context, solution) < 0:
            total_cost += self.penalty_wrong_chromosome
        else:
            total_cost += solution_checker_ga(self.context, solution)
        return -total_cost
#AA
    def __generate_chromosome(self, modele_chromosome):
        chromosome = []
        for i in modele_chromosome:
            chromosome.append(i)
        number_car = 9
        for i in range(random.randint(0, number_car)):
            chromosome.append(0)
        random.shuffle(chromosome)
        chromosome.append(0)
        chromosome.insert(0, 0)
        chromosome = str(chromosome).replace('0, 0, 0', '0').replace('0, 0', '0')
        chromosome = list(map(int, chromosome.strip('][').split(',')))
        return chromosome

#KK
    def init(self):

        self.population = [self.__generate_chromosome(self.modele_chromosome) for _ in range(self.num_population)]
        if self.best_solution and len(self.best_solution) > 0:
            self.population[-1] = self.best_solution
#KK
    def __evolution(self):  # Realize a generation, including the mating, the mutation, the elimination and the regeneration

#KKCOMPLET
        def __binary_tournement(self):  # Select certain individuals as parents by their __fitness
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

#KK
#parents:[solution,soltuon],
#child1:solution
#child2:solution

        def __pop_crossover(self, parents):  # Realize mating between parents
            for i in range(len(parents) // 2 - 1):
                parent = random.sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], parent[1])
                self.population.append(child1)
                self.population.append(child2)

            if self.best_solution:
                parent = random.sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], self.best_solution)
                self.population.append(child1)
                self.population.append(child2)

            else:
                parent = random.sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], parent[1])
                self.population.append(child1)
                self.population.append(child2)

#KKCOMPLET
        def __pop_mutation(self):  # Realize mutation for all members in the population
            population_new = copy.deepcopy(self.population)
            self.population = []
            for i in population_new:
                self.population.append(self.__chromosome_mutation(i, self.rate_mutation))

#AA
        # def __optimize(self):
        #     population_opt = copy.deepcopy(self.population)
        #     self.population = []
        #     for i in population_opt:
        #         string = str(i).replace('0, 0, 0', '0').replace('0, 0', '0')
        #         i = list(map(int, string.strip('][').split(',')))
        #         self.population.append(i)


#KKCOMPLET
#log: add best_solutions into the attributs of the GA class

        def __eliminate(self):  # Eliminate the less strong half of the population
            list_fitness = []
            for chromosome in self.population:
                list_fitness.append(self.__fitness(chromosome))
            critere = statistics.median(list_fitness)
            best_performance = max(list_fitness)

            for chromosome in self.population:
                if self.__fitness(chromosome) == best_performance:
                    self.best_solution = chromosome
                    if str(chromosome) not in self.best_solutions:
                        self.best_solutions[str(chromosome)] = self.__fitness(chromosome)
            while (len(self.population) > self.num_population):
                for chromosome in self.population:
                    if self.__fitness(chromosome)<= critere:
                        self.population.remove(chromosome)

#KKCOMPLET
#log think about making the list_fitness an attibut
        def __regeneration(self):  # Generate new-borns to maintain the number of population remains stable

            if len(self.population)<self.num_population:
                while(len(self.population)<self.num_population):
                    self.population.append(self.__generate_chromosome(self.modele_chromosome))
            
            else:
                list_fitness = []
                for chromosome in self.population:
                    list_fitness.append(self.__fitness(chromosome))

                critere = sorted(list_fitness, reverse=True)[self.num_population - 1]

                for chromosome in self.population:
                    if self.__fitness(chromosome) <= critere:
                        self.population.remove(chromosome)

                for _ in range(self.num_population - len(self.population)):
                    self.population.append(self.__generate_chromosome(self.modele_chromosome))
#KK
        parents = __binary_tournement(self)
        __pop_crossover(self, parents)
        __pop_mutation(self)
        # __optimize(self)
        __eliminate(self)
        __regeneration(self)