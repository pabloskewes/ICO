import numpy as np
from random import random

from problem import Neighborhood
from base_metaheuristic import BaseMetaheuristic


class SimulatedAnnealing(BaseMetaheuristic):
    def __init__(self, t0, cooling_factor=0.9, max_cycle_iter=100, params=None):
        super().__init__()
        self.t0 = t0
        self.cooling_factor = cooling_factor
        self.max_cycle_iter = max_cycle_iter
        self.neighborhood_params = neighborhood_params
        self.params = params

    def search(self):
        # initial setup
        N = self.neighborhood
        N.set_params(self.neighborhood_params)
        init_sol = N.initial_solution()
        best_sol = init_sol
        actual_sol = init_sol
        n_iter = 0
        new_cycle = True
        t = self.t0
        # cost = generate_cost_function(vrptw=vrptw, omega=1000)

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
                if cost(actual_sol) < cost(best_sol):
                    best_sol = actual_sol
                t = self.cooling_factor * t
        self.best_solution = best_sol

    @staticmethod
    def simulated_annealing(problem, sol, t0, cooling_factor, max_cycle_iter):
        # initial setup
        best_sol = sol
        actual_sol = sol
        n_iter = 0
        new_cycle = True
        t = t0
        N = Neighborhood(vrptw)

        # begin of cycle
        while new_cycle:
            iter_cycle = 0
            new_cycle = False
            # entropy of the neighbors calculation
            while iter_cycle < max_cycle_iter:
                iter_cycle += 1
                n_iter += 1
                neighbor = N.shuffle(actual_sol)
                dc = cost(neighbor) - cost(actual_sol)
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
                if cost(actual_sol) < cost(best_sol):
                    best_sol = actual_sol
                t = cooling_factor * t
        return best_sol



