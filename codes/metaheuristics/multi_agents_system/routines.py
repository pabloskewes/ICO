from __future__ import annotations
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from random import random
from numpy import exp

from ..tabu_search import TabuList

if TYPE_CHECKING:
    from ..base_problem import Solution
    from .base_agent import BaseAgent


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
        print('reset routine')

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


class VariableNeighborhoodDescentRoutine(Routine):
    """ Routine of VNS metaheuristic that can be done in separate iterations """
    def __init__(self, agent: BaseAgent):
        super().__init__(agent=agent)
        self.N = self.agent.N
        self.k_neighborhood = 1
        self.k_max = len(self.N.use_methods)
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
