from .base_metaheuristic import BaseMetaheuristic
from .base_problem import Solution


class VariableNeighborhoodDescent(BaseMetaheuristic):

    def __init__(self, solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

    def search(self) -> Solution:
        """
        Returns the best solution applying the Tabu Search algorithm
        :return: Solution
        """
        # Initialization of parameters
        _, N, _ = self.get_problem_components()
        solution = N.initial_solution()
        k_neighborhood = 1
        use_methods = N.use_methods
        k_max = len(use_methods)

        while k_neighborhood <= k_max:
            N.set_params({'use_methods': [use_methods[k_neighborhood-1]]})
            # We look for a neighbor
            new_solution = N.get_neighbor(solution)
            self.evolution_best_solution.append(solution.cost())
            self.evolution_explored_solutions.append(new_solution.cost())
            # If the neighbor is better than our previous solution,
            # the solution is updated and the process starts again.
            # If not, we look into the next neighborhood
            if new_solution.cost() < solution.cost():
                solution = new_solution
                k_neighborhood = 1
            else:
                k_neighborhood += 1

        return solution
