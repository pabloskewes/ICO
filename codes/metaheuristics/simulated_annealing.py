import numpy as np
from random import random
from .base_problem import Solution, Neighborhood, SolutionSpace
from .base_metaheuristic import BaseMetaheuristic


class SimulatedAnnealing(BaseMetaheuristic):
    def __init__(self, t0: int = 30, cooling_factor: float = 0.9, max_cycle_iter: int = 100,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.t0 = t0
        self.cooling_factor = cooling_factor
        self.max_cycle_iter = max_cycle_iter

        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

    def search(self) -> Solution:
        # initial setup
        _, N, _ = self.get_problem_components()
        init_sol = N.initial_solution()
        best_sol = init_sol
        self.evolution_best_solution.append(best_sol.cost())
        actual_sol = init_sol
        self.evolution_explored_solutions.append(actual_sol.cost())
        n_iter = 0
        new_cycle = True
        t = self.t0
        # begin of cycle
        while new_cycle:
            iter_cycle = 0
            new_cycle = False
            # entropy of the neighbors calculation
            while iter_cycle < self.max_cycle_iter:
                iter_cycle += 1
                n_iter += 1
                neighbor = N(actual_sol)
                dc = neighbor.cost() - actual_sol.cost()
                # if the neighbor cool down the system (less enthropy)
                # we update the best_solution
                if dc < 0:
                    actual_sol = neighbor
                    new_cycle = True
                # if not we calculate the probability
                else:
                    prob = np.exp(-dc / t)
                    q = random()
                    if q < prob:
                        actual_sol = neighbor
                        new_cycle = True
                if actual_sol.cost() < best_sol.cost():
                    best_sol = actual_sol
                self.evolution_best_solution.append(best_sol.cost())
                self.evolution_explored_solutions.append(actual_sol.cost())
                t = self.cooling_factor * t
        self.best_solution = best_sol
        return best_sol
