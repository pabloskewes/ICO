from abc import ABC, abstractmethod


class Context:
    pass


class ProblemComponent:
    valid_params = None

    def __init__(self):
        self.params = None
        self.valid_params = []

    def set_params(self, **kwargs):
        for varname, value in kwargs.items():
            if varname not in self.valid_params:
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)

    @classmethod
    def set_class_context(cls, context):
        setattr(cls, 'context', context)

    @classmethod
    def set_class_params(cls, **kwargs):
        for varname, value in kwargs.items():
            if varname not in cls.valid_params:
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(cls, varname, value)


class Solution(ABC, ProblemComponent):
    @abstractmethod
    def cost(self) -> float:
        """ The cost of a solution is defined """


class Neighborhood(ABC, ProblemComponent):
    @abstractmethod
    def initial_solution(self) -> Solution:
        """ Defines an initial solution for use in the metaheuristic process """

    @abstractmethod
    def get_neighbor(self, solution: Solution) -> Solution:
        """ The way in which the neighborhoods of a solution are chosen is defined. """

    def __call__(self, solution: Solution):
        return self.get_neighbor(solution)


class SolutionSpace(ABC, ProblemComponent):
    @abstractmethod
    def distance(self, s1: Solution, s2: Solution) -> float:
        """ Defines the distance between 2 solutions in the solution space. """


class Problem:
    context = Context
    solution = Solution
    neighborhood = Neighborhood
    solution_space = SolutionSpace




