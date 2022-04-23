import numpy as np
from random import random
from .base_problem import Solution, Neighborhood, SolutionSpace
from .base_metaheuristic import BaseMetaheuristic

from tqdm import tqdm


class SimulatedAnnealing(BaseMetaheuristic):
    def __init__(self, t0: int = 30, cooling_factor: float = 0.9, max_cycle_iter: int = 100,
                 progress_bar: bool = False, verbose: int = 0,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()
        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

        # Metaheuristic Hyperparameters
        self.t0 = t0
        self.cooling_factor = cooling_factor
        self.max_cycle_iter = max_cycle_iter
        self.progress_bar = progress_bar

    def search(self) -> Solution:
        # initial setup
        pbar = tqdm(total=self.max_cycle_iter) if self.progress_bar else None
        _, N, _ = self.get_problem_components()
        init_sol = N.initial_solution()
        best_sol = init_sol
        self.evolution_best_solution.append(best_sol.cost())
        actual_sol = init_sol
        self.evolution_explored_solutions.append(actual_sol.cost())
        n_iter = 0
        new_cycle = True
        t = self.t0
<<<<<<< HEAD
        it = 0
        pbar.set_description('Cost: %.2f' %best_sol.cost())
=======
        pbar.set_description('Cost: %.2f' % best_sol.cost())
>>>>>>> 1c197308b43bdc17860a58c65df1752a7538188a
        # begin of cycle
        while new_cycle:
            iter_cycle = 0
            new_cycle = False
            # entropy of the neighbors calculation
            while iter_cycle < self.max_cycle_iter:
                if self.progress_bar:
                    pbar.update()
                iter_cycle += 1
                n_iter += 1
                neighbor = N(actual_sol)
                self.evolution_best_solution.append(best_sol.cost())
                self.evolution_explored_solutions.append(neighbor.cost())
                dc = neighbor.cost() - actual_sol.cost()
                # if the neighbor cool down the system (less entropy)
                # we update the best_solution
                if dc < 0:
                    actual_sol = neighbor
                    new_cycle = True
                # if not we calculate the probability
                elif dc > 0:
                    prob = np.exp(-1.0 * dc / t)
                    q = random()
                    if q < prob:
                        actual_sol = neighbor
                        new_cycle = True
                if actual_sol.cost() < best_sol.cost():
                    best_sol = actual_sol
            t = self.cooling_factor * t
            if self.progress_bar:
                pbar.reset()
                pbar.set_description('Cost: %.2f' % best_sol.cost())
        if self.progress_bar:
            pbar.close()
        self.best_solution = best_sol
        return best_sol

    def iterate(self) -> Solution:
        # initial setup
        while self.new_cycle:
            self.iter_cycle = 0
            self.new_cycle = False
            # entropy of the neighbors calculation
            while self.iter_cycle < self.max_cycle_iter:
                if self.progress_bar:
                    self.pbar.update()
                self.iter_cycle += 1
                self.n_iter += 1
                self.neighbor = self.N(self.actual_sol)
                self.evolution_best_solution.append(self.best_sol.cost())
                self.evolution_explored_solutions.append(self.neighbor.cost())
                dc = self.neighbor.cost() - self.actual_sol.cost()
                # if the neighbor cool down the system (less entropy)
                # we update the best_solution
                if dc < 0:
                    self.actual_sol = self.neighbor
                    self.new_cycle = True
                # if not we calculate the probability
                elif dc > 0:
                    prob = np.exp(-1.0 * dc / self.t)
                    q = random()
                    if q < prob:
                        self.actual_sol = self.neighbor
                        self.new_cycle = True
                if self.actual_sol.cost() < self.best_sol.cost():
                    self.best_sol = self.actual_sol
                self.t *= self.cooling_factor
            if self.progress_bar:
                self.pbar.reset()
                self.pbar.set_description('Cost: %.2f' % self.best_sol.cost())
        return self.best_sol
