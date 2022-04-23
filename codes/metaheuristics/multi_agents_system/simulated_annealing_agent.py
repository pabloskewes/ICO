from random import random
from numpy import exp


class SAAgent:
    def __init__(self, t0: int = 30, cooling_factor: float = 0.9, max_iter=100, init_sol=None):
        self.t0 = t0
        self.t = self.t0
        self.cooling_factor = cooling_factor
        self.max_iter = max_iter
        self.n_iter = 0
        self.is_finished = False
        self.init_sol = init_sol
        self.actual_sol = self.init_sol
        self.best_sol = self.init_sol
        self.new_cycle = False

    def reset_params(self):
        self.t = self.t0
        self.is_finished = False

    def explore(self, neighborhood):
        if not self.is_finished:
            self.n_iter += 1
            neighbor = neighborhood(self.actual_sol)
            dc = neighbor.cost() - self.best_sol.cost()
            # if the neighbor cool down the system (less entropy)
            # we update the best_solution
            if dc < 0:
                self.actual_sol = neighbor
                self.new_cycle = True
            # if not we calculate the probability
            if dc > 0:
                prob = exp(-1.0* dc / self.t)
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
