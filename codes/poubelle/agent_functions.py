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


class TabuAgent:
    def __init__(self, max_tabu: int = 10, max_iter: int = 100, tabu_mode: str = 'default'):
        self.max_tabu = max_tabu
        self.tabu_mode = tabu_mode
        self.T = TabuList(mode=self.tabu_mode)
        self.max_iter = max_iter
        self.last_solution = None

    def explore(self, neighborhood, init_sol=None):
        if init_sol is None:
            init_sol = neighborhood.initial_solution()

        if self.last_solution is None:
            self.last_solution = init_sol

        self.T.push(init_sol)
        new_solution = neighborhood(init_sol)
        n_cycle = 0
        while self.T.contains(new_solution):
            new_solution = neighborhood(init_sol)
            if n_cycle == self.max_iter:
                new_solution = self.last_solution
                break
            n_cycle += 1
        if new_solution.cost() < self.last_solution.cost():
            self.last_solution = new_solution

        return new_solution
