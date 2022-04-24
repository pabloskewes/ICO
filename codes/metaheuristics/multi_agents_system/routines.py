from random import random
from numpy import exp
from typing import Optional
from abc import ABC, abstractmethod

from ..tabu_search import TabuList
from ..base_problem import Solution, Neighborhood
from .base_agent import BaseAgent


class BaseRoutine:
    """  Base routine of a metaheuristic that can be done iteration by iteration. """
    def __init__(self, init_sol: Solution = None):
        self.init_sol = init_sol
        self.agent: Optional[BaseAgent] = None
        self.is_finished = False
        self.best_sol = self.init_sol

    def fit_agent(self, agent: BaseAgent):
        self.agent = agent

    def reset_routine(self) -> None:
        """ Resets the parameters in the memory of the metaheuristic. """
        print('reset routine')

    def iteration(self) -> Solution:
        """ Performs an iteration of the metaheuristic and returns the solution found. """
        neighbor = self.agent.explore(self.best_sol)
        return neighbor

class SimulatedAnnealingRoutine(BaseRoutine):
    """ Routine of Simulated Annealing metaheuristic that can be done in separate iterations """
    def __init__(self, t0: int = 30, cooling_factor: float = 0.9, max_iter=100, init_sol: Solution = None):
        super(BaseRoutine).__init__(init_sol=init_sol)

        self.t0 = t0
        self.t = self.t0
        self.cooling_factor = cooling_factor
        self.max_iter = max_iter
        self.n_iter = 0
        self.init_sol = init_sol
        self.actual_sol = self.init_sol
        self.best_sol = self.init_sol
        self.new_cycle = False

    def reset_routine(self):
        self.t = self.t0
        self.is_finished = False
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


class TabuRoutine(BaseRoutine):
    """ Routine of Tabu Search metaheuristic that can be done in separate iterations """
    def __init__(self, max_tabu: int = 100, max_iter: int = 100, tabu_mode: str = 'default', init_sol: Solution = None):
        super(BaseRoutine).__init__(init_sol=init_sol)

        self.max_tabu = max_tabu
        self.T = TabuList(mode=tabu_mode)
        self.max_iter = max_iter
        self.init_sol = init_sol
        self.actual_sol = self.init_sol
        self.last_visited_sol = self.init_sol
        self.best_sol = self.init_sol
        self.n_iter = 0
        self.best_iter = 0
        self.T.push(self.init_sol)

    def reset_routine(self):
        self.is_finished = False
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


class VariableNeighborhoodDescentRoutine(BaseRoutine):
    """ Routine of VNS metaheuristic that can be done in separate iterations """
    def __init__(self, neighborhood, init_sol=None):
        super(BaseRoutine).__init__(init_sol=init_sol)
        self.N = neighborhood
        self.k_neighborhood = 1
        self.k_max = len(self.N.use_methods)
        self.init_sol = self.N.initial_solution() if init_sol is None else init_sol
        self.best_sol = self.init_sol

    def reset_routine(self):
        self.k_neighborhood = 1
        self.best_sol = self.init_sol
        self.is_finished = False

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
