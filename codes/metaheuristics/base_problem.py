from abc import ABC, abstractmethod


class Context:
    pass


class Solution(ABC):
    def __init__(self):
        self.params = None
        self.context = None
        self.valid_params = []

    def set_params(self, **kwargs):
        for varname, value in kwargs.items():
            if varname not in self.valid_params:
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)

    @abstractmethod
    def cost(self) -> float:
        """ The cost of a solution is defined """


class Neighborhood(ABC):
    def __init__(self):
        self.params = None
        self.context = None
        self.valid_params = []

    def set_params(self, **kwargs):
        for varname, value in kwargs.items():
            if varname not in self.valid_params:
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)

    @abstractmethod
    def initial_solution(self) -> Solution:
        """ Defines an initial solution for use in the metaheuristic process """

    @abstractmethod
    def get_neighbor(self, solution: Solution) -> Solution:
        """ The way in which the neighborhoods of a solution are chosen is defined. """

    def __call__(self, solution: Solution):
        return self.get_neighbor(solution)


class SolutionSpace(ABC):
    def __init__(self):
        self.params = None
        self.context = None
        self.valid_params = []

    def set_params(self, **kwargs):
        for varname, value in kwargs.items():
            if varname not in self.valid_params:
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)


class Problem:
    def __init__(self, context: Context, solution: Solution, neighborhood: Neighborhood, solution_space: SolutionSpace):
        self.context = context
        self.solution = solution
        self.neighborhood = neighborhood
        self.solution_space = solution_space
        # context is set for every attribute
        self.solution.context = context
        self.neighborhood.context = context
        self.solution_space.context = context
