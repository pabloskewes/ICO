import random
import re
import statistics
import pandas as pd
import os
import sys
from math import sqrt
import copy
import numpy as np
from .base_problem import Solution, Neighborhood, SolutionSpace
from .base_metaheuristic import BaseMetaheuristic

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
AA               Methode that should be rewrite in the outside vlasses
LOG              log of the modification of the main structure
'''
# TDL:

# ga=GeneticAlgorithm()
# ga.context=load_solomon()
# from solution import sol_to_list_routes

'''
LOG
---------------------------------------------------
1.      Cancel the optimize part
2.      The __chromosome_crossover should be implemented in the outside, but i don't know where for now
3.      Realize the solution_checker_GA in the cost() like cost_ga() or an override of cost(), or an override of the solution_check()
'''


class GeneticAlgorithm(BaseMetaheuristic):
    def __init__(self,num_evolu_per_search=10, num_parent=4, num_population=20, rate_mutation=0.2,
                 population=[], solution_params=None, neighborhood_params=None, solution_space_params=None):
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
        

    # KK
    def __get_best_solution(self):
        if self.best_solution:
            return self.best_solution
        else:
            print("Empty solution. To get the best solution, use search() first")
            return False

    # KK
    def __chromosome_mutation(self, chromosome, prob):
        if random.random() < prob:
            N=self.NEIGHBORHOOD() 
            N.set_params({'choose_mode':'reverse_a_sequence'})
            return N(chromosome)
        else:
            return chromosome

    # KK
    def __chromosome_crossover(self, parent1, parent2):  # what about cross-mute?
        N=self.NEIGHBORHOOD()
        N.set_params({'choose_mode':'crossover'})
        return N.get_neighbor_from_two(parent1, parent2)

    def search(self):
        """ Performs metaheuristic search """

        for _ in range(self.num_evolu_per_search):
            self.__evolution()
            
        self.evolution_best_solution.append(-self.__fitness(self.best_solution))
        self.evolution_explored_solutions=self.evolution_explored_solutions[:len(self.evolution_best_solution)]
        return self.best_solution

    def __fitness(self, solution):
        """
        is the fitness always has the same definition with the cost of the solution?
        """

        total_cost =solution.cost()
        penalty=0

        if solution.cost()==0:
            penalty+=float('inf')

        all_customers_check = solution.all_customers_checker()
        not_repeated_customers_check = solution.not_repeated_customers_checker()

        for route in solution.routes:
            penalty+=int(not solution.route_checker(route))*self.penalty_wrong_chromosome

        penalty += int(not all_customers_check)*self.penalty_wrong_chromosome
        penalty += int(not not_repeated_customers_check)*self.penalty_wrong_chromosome
        return -total_cost-penalty

    # KK
    def __generate_chromosome(self):

        _, N, _ = self.get_problem_components()
        chromosome = N.initial_solution()

        return chromosome

    # KK
    def init(self):

        self.population = [self.__generate_chromosome() for _ in range(self.num_population)]
        # might need a good seed to start well

    # KK
    def __evolution(self): 

        # KKCOMPLET
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

        # KKCOMPLET
        def __pop_mutation(self):  # Realize mutation for all members in the population
            population_new = copy.deepcopy(self.population)
            self.population = []
            for i in population_new:
                self.population.append(self.__chromosome_mutation(i, self.rate_mutation))

        # AA
        # def __optimize(self):
        #     population_opt = copy.deepcopy(self.population)
        #     self.population = []
        #     for i in population_opt:
        #         string = str(i).replace('0, 0, 0', '0').replace('0, 0', '0')
        #         i = list(map(int, string.strip('][').split(',')))
        #         self.population.append(i)

        # KKCOMPLET
        # log: save the best solution in which form?    

        def __eliminate(self):  # Eliminate the less strong half of the population
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

        # KKCOMPLET
        # log think about making the list_fitness an attibut
        def __regeneration(self):  # Generate new-borns to maintain the number of population remains stable

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

        # KK
        parents = __binary_tournement(self)
        __pop_crossover(self, parents)
        __pop_mutation(self)
        # __optimize(self)
        __eliminate(self)
        __regeneration(self)
