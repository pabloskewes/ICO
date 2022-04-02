from abc import ABC, abstractmethod
from types import Dict


class Context:
    def __init__(self):
        pass


class Solution(ABC):
    def __init__(self):
        self.params = None

    def set_params(self, params):
        self.params = params

    @abstractmethod
    def cost(self) -> float:
        """ The cost of a solution is defined """


class SolutionSpace(ABC):
    def __init__(self):
        self.params = None

    def set_params(self, params):
        self.params = params


class Neighborhood(ABC):
    def __init__(self):
        self.params = None

    def set_params(self, params):
        self.params = params

    @abstractmethod
    def initial_solution(self) -> Solution:
        """ Defines an initial solution for use in the metaheuristic process """

    @abstractmethod
    def get_neighbor(self, solution: Solution, *args, **kwargs) -> Solution:
        """ The way in which the neighborhoods of a solution are chosen is defined. """

    def __call__(self, solution: Solution, neigh_params: Dict = None):
        if neigh_params is not None:
            return self.get_neighbors(solution, neigh_params)
        else:
            return self.get_neighbors(solution)


class Problem:
    def __init__(self, context: Context, solution: Solution, neighborhood: Neighborhood, solution_space: SolutionSpace):
        self.solution = solution
        self.neighborhood = neighborhood
        self.solution_space = solution_space


# class ProblemFactory:
#     def __init__(self):
#         pass
#
#     def generate_problem(self, solution_params, neigh_params, space_params) -> Problem:
#         solution = Solution()
#         solution.set_params(solution_params)
#         neighborhood = Neighborhood()
#         neighborhood.set_params(neigh_params)
#         space = SolutionSpace()
#         space.set_params(space_params)
#         return Problem(solution=solution, neighborhood=neighborhood, solution_space=space)





