import numpy as np
from random import random

from .solution import generate_cost_function
from neighborhood import Neighborhood
from MetaAlgorithm import MetaAlgorithm

class RSAlgorithm(MetaAlgorithm):
    
    def __init__(self, t0, cooling_factor=0.9, max_cycle_iter=100):
        super().__init__()
        self.t0 = t0
        self.cooling_factor = cooling_factor
        self.max_cycle_iter = max_cycle_iter

    def search(self, vrptw, sol):
        # initial setup
        best_sol = sol
        actual_sol = sol
        n_iter = 0
        new_cycle = True
        T = self.t0
        cost = generate_cost_function(vrptw=vrptw, omega=1000)
        N = Neighborhood(vrptw)

        # begin of cycle
        while new_cycle:
            iter_cycle = 0
            new_cycle = False
            # entropy of the neighbors calculation
            while iter_cycle < self.max_cycle_iter:
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
                    prob = np.exp(-dc / T)
                    q = random()
                    if q < prob:
                        actual_sol = neighbor
                        new_cycle = True
                if cost(actual_sol) < cost(best_sol):
                    best_sol = actual_sol
                T = self.cooling_factor * T
        self.best_solution = best_sol




