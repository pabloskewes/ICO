from __future__ import annotations
from typing import Dict, Any, Optional, List, Type, TYPE_CHECKING
from random import random, sample
from numpy import exp
from copy import deepcopy
from statistics import median

from ..tabu_search import TabuList

if TYPE_CHECKING:
    from ..base_problem import Solution
    from .base_agent import BaseAgent
    from ..base_metaheuristic import BaseMetaheuristic


class Routine:
    """  Base routine of a metaheuristic that can be done iteration by iteration. """
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.init_sol = self.agent.in_solution
        self.is_finished = False
        self.best_sol = self.init_sol
        self.hyperparameters: List[str] = []

    def reset_routine(self) -> None:
        """ Resets the parameters in the memory of the metaheuristic. """
        self.is_finished = False
        self.init_sol = self.agent.in_solution

    def iteration(self) -> Solution:
        """ Performs an iteration of the metaheuristic and returns the solution found. """
        neighbor = self.agent.explore(self.best_sol)
        return neighbor

    def set_params(self, params: Dict[str, Any]) -> None:
        if params is None:
            return
        """ Set parameters of routine """
        for varname, value in params.items():
            if varname not in self.hyperparameters:
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)


class MetaheuristicRoutine(Routine):
    """ Routine that applies a complete metaheuristic at each iteration of the agent """
    def __init__(self, agent: BaseAgent):
        super().__init__(agent=agent)

        self.metaheuristic_params: Dict[str, Any] = {}
        self.METAHEURISTIC: Optional[Type[BaseMetaheuristic]] = None
        # noinspection PyArgumentList
        self.metaheuristic = self.METAHEURISTIC(**self.metaheuristic_params)

    def reset_routine(self) -> None:
        # noinspection PyArgumentList
        self.metaheuristic = self.METAHEURISTIC(**self.metaheuristic_params)

    def iteration(self) -> Solution:
        problem = self.agent.model.problem
        new_sol = self.metaheuristic.fit_search(problem)
        return new_sol


class SimulatedAnnealingRoutine(Routine):
    """ Routine of Simulated Annealing metaheuristic that can be done in separate iterations """
    def __init__(self, agent: BaseAgent):
        super().__init__(agent=agent)

        self.t0: int = 30
        self.cooling_factor: float = 0.9
        self.max_iter: int = 100

        self.t = self.t0
        self.n_iter = 0
        self.actual_sol = self.init_sol
        self.best_sol = self.init_sol
        self.new_cycle = False

        self.hyperparameters = ['t0', 'cooling_factor', 'max_iter']

    def reset_routine(self):
        super().reset_routine()
        self.t = self.t0
        self.actual_sol = self.init_sol
        self.best_sol = self.init_sol
        self.n_iter = 0
        self.new_cycle = False

    def iteration(self) -> Solution:
        if not self.is_finished:
            self.n_iter += 1
            neighbor = self.agent.explore(self.actual_sol)
            dc = neighbor.cost() - self.best_sol.cost()
            # if the neighbor cool down the system (less entropy)
            # we update the best_solution
            if dc < 0:
                self.actual_sol = neighbor
                self.new_cycle = True
            # if not we calculate the probability
            if dc > 0:
                prob = exp(-1.0 * dc / self.t)
                q = random()
                if q < prob:
                    self.actual_sol = neighbor
                    self.new_cycle = True

            if self.actual_sol.cost() < self.best_sol.cost():
                self.best_sol = self.actual_sol

            if self.n_iter >= self.max_iter:
                self.n_iter = 0
                self.t *= self.cooling_factor
                if not self.new_cycle:
                    self.is_finished = True
                self.new_cycle = False

        return self.best_sol


class TabuRoutine(Routine):
    """ Routine of Tabu Search metaheuristic that can be done in separate iterations """
    def __init__(self, agent: BaseAgent):
        super().__init__(agent=agent)

        self.max_tabu: int = 100
        self.max_iter: int = 100
        self.tabu_mode: str = 'default'

        self.T = TabuList(mode=self.tabu_mode)
        self.actual_sol = self.init_sol
        self.last_visited_sol = self.init_sol
        self.best_sol = self.init_sol
        self.n_iter = 0
        self.best_iter = 0
        self.T.push(self.init_sol)

        self.hyperparameters = ['max_tabu', 'max_iter', 'tabu_mode']

    def reset_routine(self):
        super().reset_routine()
        self.T.empty()
        self.T.push(self.init_sol)
        self.last_visited_sol = self.init_sol
        self.best_sol = self.init_sol
        self.actual_sol = self.init_sol
        self.n_iter = 0
        self.best_iter = 0

    def iteration(self) -> Solution:
        if not self.is_finished:
            self.n_iter += 1
            new_solution = self.agent.explore(self.actual_sol)

            n_cycle = 0
            while self.T.contains(new_solution):
                new_solution = self.agent.N(self.actual_sol)
                if n_cycle == self.max_iter:
                    self.T.push(self.actual_sol)
                    self.actual_sol = self.last_visited_sol
                    return self.best_sol
                n_cycle += 1

            if new_solution.cost() < self.best_sol.cost():
                self.last_visited_sol = self.best_sol
                self.best_sol = new_solution
                self.best_iter = self.n_iter

            self.T.push(new_solution)
            self.actual_sol = new_solution

            if (self.n_iter - self.best_iter) >= self.max_iter:
                self.is_finished = True

        return self.best_sol


class GeneticRoutine(Routine):
    """ Routine of Genetic Algorithm that can be done in separate iterations """
    def __init__(self, agent: BaseAgent):
        super().__init__(agent=agent)
        self.num_evolu_per_search = 10
        self.population = []
        self.rate_mutation = 0.2
        self.num_parent = 4
        self.num_population = 20
        self.penalty_wrong_chromosome = 40000
        self.has_init = False
        self.list_nation = []
        self.threshold = 10
        self.reproductive_isolation = False
        self.best_seed = True
        self.n_iter = 0

    def reset_routine(self):
        super().reset_routine()
        self.has_init = False
        self.n_iter = 0
        self.population = []
        self.list_nation = []
        self.reproductive_isolation = False
        self.best_seed = True

    def iteration(self) -> Solution:
        if not self.is_finished:
            if not self.has_init:
                self.__init()

            self.n_iter += 1
            self.__evolution()

            if self.n_iter == self.num_evolu_per_search:
                self.is_finished = True

        return self.best_sol

    def __chromosome_mutation(self, chromosome, prob):
        if random() < prob and chromosome.cost() > 0:
            return self.agent.explore(chromosome)
        else:
            return chromosome

    def __chromosome_crossover(self, parent1, parent2):

        N = deepcopy(self.agent.N)
        N.set_params({'use_methods':'crossover'})

        SP=self.agent.problem.solution_space()
        if SP.distance(parent1, parent2)<self.threshold and self.reproductive_isolation==True:
            return N.get_neighbor_from_two(parent1, parent2)
        else:
            return self.__generate_chromosome(),self.__generate_chromosome()

    def __fitness(self, solution):
        return -solution.cost()

    def __generate_chromosome(self):
        return self.init_sol

    def __init(self):
        self.has_init=True

        while(len(self.population)<self.num_population):

            new_born=self.__generate_chromosome()
            if new_born.cost()!=0:
                self.population.append(new_born)

    def __evolution(self):

        def __binary_tournement(self):
            parents = []
            for i in range(self.num_parent):
                candidate = sample(self.population, 2)
                if self.__fitness(candidate[0]) > self.__fitness(candidate[1]):
                    if random() < 0.95:
                        parents.append(candidate[0])
                    else:
                        parents.append(candidate[1])
                else:
                    if random() < 0.95:
                        parents.append(candidate[1])
                    else:
                        parents.append(candidate[0])

            return parents

        def __pop_crossover(self, parents):
            for i in range(len(parents) // 2 - 1):
                parent = sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], parent[1])

                self.population.append(child1)
                self.population.append(child2)

            if self.best_sol and self.best_seed:
                parent = sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], self.best_sol)

            else:
                parent = sample(parents, 2)
                child1, child2 = self.__chromosome_crossover(parent[0], parent[1])

            self.population.append(child1)
            self.population.append(child2)

        def __pop_mutation(self):
            population_new = deepcopy(self.population)
            self.population = []
            for i in population_new:
                self.population.append(self.__chromosome_mutation(i, self.rate_mutation))

        def __eliminate(self):
            list_fitness = []
            for chromosome in self.population:
                fitness=self.__fitness(chromosome)
                list_fitness.append(fitness)
            critere = median(list_fitness)
            best_performance = max(list_fitness)

            for chromosome in self.population:
                if self.__fitness(chromosome) == best_performance:
                    self.best_sol = chromosome


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
        for chromosome in self.population:
            if not chromosome.checker():
                print('err: crossover')
        __pop_mutation(self)
        for chromosome in self.population:
            if not chromosome.checker():
                print('err: mut')
        __eliminate(self)
        for chromosome in self.population:
            if not chromosome.checker():
                print('err: eli')
        __regeneration(self)
        for chromosome in self.population:
            if not chromosome.checker():
                print('err: reg')


class VariableNeighborhoodDescentRoutine(Routine):
    """ Routine of VNS metaheuristic that can be done in separate iterations """
    def __init__(self, agent: BaseAgent):
        super().__init__(agent=agent)
        self.N = self.agent.N
        self.k_neighborhood = 1
        self.k_max = len(self.N.use_methods)
        self.best_sol = self.init_sol

    def reset_routine(self):
        super().reset_routine()
        self.k_neighborhood = 1
        self.best_sol = self.init_sol

    def iteration(self) -> Solution:
        if not self.is_finished:
            self.N.set_params({'choose_mode': self.k_neighborhood})
            new_solution = self.N(self.best_sol)
            if new_solution.cost() < self.best_sol.cost():
                self.best_sol = new_solution
                self.k_neighborhood = 1
            else:
                self.k_neighborhood += 1
            self.is_finished = self.k_neighborhood == self.k_max

        return self.best_sol
