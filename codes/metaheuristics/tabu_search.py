from .base_metaheuristic import BaseMetaheuristic
from .base_problem import Solution


class TabuSearch(BaseMetaheuristic):
    """
    Class inherited from BaseMetaheuristic that implements the Tabu Search algorithm
    """
    class _TabuList:
        """
        FIFO list that represents the tabu list
        """
        def __init__(self):
            self.tabu_list = []
            self.size = 0

        def push(self, e):
            self.tabu_list.append(e)
            self.size += 1

        def remove_first(self):
            self.tabu_list = self.tabu_list[1:]
            self.size -= 1

        def contains(self, e):
            return e in self.tabu_list

    def __init__(self, lower_bound=100, max_iter=100, max_tabu=10, solution_params=None,
                 neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.actual_solution = None
        self.lower_bound = lower_bound
        self.max_iter = max_iter
        self.max_tabu = max_tabu
        
        self.params = {'solution' : solution_params, 'neighborhood' : neighborhood, 
                       'solution_space' : solution_space_params}
        
    def search(self) -> Solution:
        """
        Returns the best solution applying the Tabu Search algorithm
        :return: Solution
        """
        # Initialization of parameters
        n_iter = 0
        T = TabuSearch._TabuList()
        _, N, _ = self.get_problem_components()
        self.best_solution = N.initial_solution()
        self.actual_solution = self.best_solution
        self.evolution_best_solution.append(self.best_solution.cost())
        self.evolution_explored_solutions.append(self.actual_solution.cost())
        T.push(self.actual_solution)
        best_iter = 0

        # Stop conditions : cost under a minimal expected cost or maximum iteration reached
        while (self.best_solution.cost() > self.lower_bound) and ((n_iter - best_iter) < self.max_iter):
            n_iter += 1

            # Generates a new solution from the neighborhood of the actual solution
            new_solution = N(self.actual_solution)

            # If the solution is in the Tabu list, another one is generated
            while T.contains(new_solution):
                new_solution = N(self.actual_solution)

            # If the new solution cost is less than the actual best solution cost, the best solution is updated
            if self.best_solution.cost() > new_solution.cost():
                self.best_solution = new_solution
                best_iter = n_iter

            # The solution found is added to the Tabu list, and the actual solution is updated with the
            # new solution
            T.push(new_solution)
            self.actual_solution = new_solution

            # Aspiration : if the size of the Tabu list is greater than the maximum given, the first solution
            # added is withdrawn
            if T.size > self.max_tabu:
                T.remove_first()

            self.evolution_explored_solutions.append(self.actual_solution.cost())
            self.evolution_best_solution.append(self.best_solution.cost())
        return self.best_solution
