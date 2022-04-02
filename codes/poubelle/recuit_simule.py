import numpy as np
from random import random

from solution import generate_cost_function
from neighborhood import Neighborhood


def simulated_annealing(vrptw, sol, T0, cooling_factor=0.9, max_cycle_iter=100):
    # initial setup
    best_sol = sol
    actual_sol = sol
    n_iter = 0
    new_cycle = True
    T = T0
    cost = generate_cost_function(vrptw=vrptw, omega=1000)
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
                prob = np.exp(-dc/T)
                q = random()
                if q < prob:
                    actual_sol = neighbor
                    new_cycle = True
            if cost(actual_sol) < cost(best_sol):
                best_sol = actual_sol
            T = cooling_factor*T
    return best_sol

