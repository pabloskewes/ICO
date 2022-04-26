from tqdm import tqdm

from .base_metaheuristic import BaseMetaheuristic
from .base_problem import Solution


class GreedySearch(BaseMetaheuristic):
    def __init__(self, max_cycle_iter: int = 100, progress_bar=False, verbose: int = 0,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()
        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

        # Metaheuristic Hyperparameters
        self.max_cycle_iter = max_cycle_iter
        self.progress_bar = progress_bar
        self.verbose = verbose

    def search(self) -> Solution:
        pbar = tqdm(total=self.max_cycle_iter) if self.progress_bar else None
        _, N, _ = self.get_problem_components()
        self.best_solution = N.initial_solution()
        self.evolution_best_solution.append(self.best_solution.cost())
        if self.progress_bar:
            pbar.set_description('Cost: %.2f' % self.best_solution.cost())
        n = 0
        while n < self.max_cycle_iter:
            n += 1
            if self.progress_bar:
                pbar.update(n)
            new_sol = N(self.best_solution)
            if new_sol.cost() < self.best_solution.cost():
                self.best_solution = new_sol
                self.evolution_best_solution.append(self.best_solution.cost())
                n = 0
                if self.progress_bar:
                    pbar.reset()
                    pbar.set_description('Cost: %.2f' % self.best_solution.cost())
                continue
        if self.progress_bar:
            pbar.reset()
            pbar.set_description('Cost: %.2f' % self.best_solution.cost())
            pbar.close()

        return self.best_solution
