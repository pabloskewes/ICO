from .base_metaheuristic import BaseMetaheuristic
from .base_problem import Solution
from random import randint

from tqdm import tqdm


class TabuList:
    """
    FIFO list that represents the tabu list
    """
    def __init__(self, mode: str = 'default'):
        self.tabu_list = []
        self.size = 0
        self.mode = mode

    def push(self, e):
        if not self.contains(e):
            self.tabu_list.append(e)
            self.size += 1

    def remove_first(self):
        self.tabu_list = self.tabu_list[1:]
        self.size -= 1

    def contains_default(self, e):
        return e in self.tabu_list

    def contains_customs(self, e):
        return any(e == e2 for e2 in self.tabu_list)

    def contains(self, e):
        if self.mode == 'default':
            return self.contains_default(e)
        elif self.mode == 'custom':
            return self.contains_customs(e)
        else:
            raise Exception(f'{self.mode} is not a valid mode for tabu list')


class TabuSearch(BaseMetaheuristic):
    """
    Class inherited from BaseMetaheuristic that implements the Tabu Search algorithm
    """
    def __init__(self, lower_bound: int = 100, max_iter: int = 100, max_tabu: int = 100,
                 tabu_mode: str = 'default', progress_bar: bool = False,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.actual_solution = None
        self.last_visited_solution = None
        self.lower_bound = lower_bound
        self.max_iter = max_iter
        self.max_tabu = max_tabu
        self.tabu_mode = tabu_mode
        self.progress_bar = progress_bar

        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

    def search(self) -> Solution:
        """
        Returns the best solution applying the Tabu Search algorithm
        :return: Solution
        """
        # Initialization of parameters
        pbar = tqdm(total=self.max_iter) if self.progress_bar else None
        n_iter = 0
        T = TabuList(mode=self.tabu_mode)
        _, N, _ = self.get_problem_components()
        self.best_solution = N.initial_solution()
        self.actual_solution = self.best_solution
        self.last_visited_solution = self.best_solution
        self.evolution_best_solution.append(self.best_solution.cost())
        self.evolution_explored_solutions.append(self.actual_solution.cost())
        T.push(self.actual_solution)
        best_iter = 0
        pbar.set_description('Cost: %.2f' % self.best_solution.cost())

        # Stop conditions : cost under a minimal expected cost or maximum iteration reached
        while (self.best_solution.cost() > self.lower_bound) and ((n_iter - best_iter) < self.max_iter):
            n_iter += 1
            if self.progress_bar:
                pbar.update()


            # Generates a new solution from the neighborhood of the actual solution
            new_solution = N(self.actual_solution)
            self.evolution_explored_solutions.append(new_solution.cost())
            self.evolution_best_solution.append(self.best_solution.cost())

            # If the solution is in the Tabu list, another one is generated
            n_cycle = 0
            aspiration_condition = False
            while T.contains(new_solution):
                new_solution = N(self.actual_solution)
                self.evolution_explored_solutions.append(new_solution.cost())
                self.evolution_best_solution.append(self.best_solution.cost())
                if n_cycle == self.max_iter:
                    T.push(self.actual_solution)
                    self.actual_solution = self.last_visited_solution
                    aspiration_condition = True
                    break
                n_cycle += 1
            if aspiration_condition:
                continue

            # If the new solution cost is less than the actual best solution cost, the best solution is updated
            if self.best_solution.cost() > new_solution.cost():
                if self.progress_bar:
                    pbar.reset()
                    pbar.set_description('Cost: %.2f' % new_solution.cost())
                self.last_visited_solution = self.best_solution
                self.best_solution = new_solution
                best_iter = n_iter

            # The solution found is added to the Tabu list, and the actual solution is updated with the new solution
            T.push(new_solution)
            self.actual_solution = new_solution

            # Aspiration : if the size of the Tabu list is greater than the maximum given, the first solution
            # added is withdrawn
            if T.size > self.max_tabu:
                T.remove_first()

        if self.progress_bar:
            pbar.close()

        return self.best_solution
