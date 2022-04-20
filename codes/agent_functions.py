import numpy as np
from random import random
from metaheuristics.tabu_search import TabuList
from tqdm import tqdm

class SimulatedAnnealingAgent:

    def __init__(self, t0: int = 30, cooling_factor: float = 0.9):
        self.t = t0
        self.cooling_factor = cooling_factor

    def explore(self, neighborhood, init_sol=None):
        if init_sol is None:
            init_sol = neighborhood.initial_solution()

        best_sol = init_sol
        # self.evolution_best_solution.append(best_sol.cost())
        # self.evolution_explored_solutions.append(actual_sol.cost())
        neighbor = neighborhood(best_sol)
        dc = neighbor.cost() - best_sol.cost()
        # if the neighbor cool down the system (less entropy)
        # we update the best_solution
        best_sol = neighbor if dc < 0 else best_sol
        # if not we calculate the probability
        if dc > 0:
            prob = np.exp(-1.0* dc / self.t)
            q = random()
            best_sol = neighbor if q < prob else best_sol
        self.t *= self.cooling_factor
        return best_sol
