from abc import ABC, abstractmethod
from .base_problem import Solution, Neighborhood, SolutionSpace, Problem
from copy import deepcopy


class BaseMetaheuristic(ABC):
    """
    Abstract class inherited from ABC (Abstract Base Class) class.
    It is used to create different algorithm classes.
    """
    def __init__(self):
        self.best_solution = None
        self.Solution = None
        self.neighborhood = None
        self.solution_space = None
        self.problem = None
        self.cost_list = []
        self.params = {'solution': dict(), 'neighborhood': dict(), 'solution_space': dict()}

    def fit(self, problem: Problem):
        """ Fits a metaheuristic algorithm to a specific problem """
        # problem_copy = deepcopy(problem)
        # TODO: Determine if it's a good idea to give empty solution instance to base metaheuristic
        context = problem.context

        # instantiation of problem components
        solution_class = problem.solution
        setattr(solution_class, 'context', context)
        neighborhood = problem.neighborhood(context=context)
        # solution_space = problem.solution_space(context=context)

        # setting context in every problem component
        # solution.set_context(context)
        # neighborhood.set_context(context)
        # solution_space.set_context(context)

        # setting custom params in every problem component
        if self.params is not None:
            if self.params['solution'] is not None:
                # solution.set_params(self.params['solution'])
                solution_class.set_class_params(self.params['solution'])
            if self.params['neighborhood'] is not None:
                neighborhood.set_params(**self.params['neighborhood'])
            # if self.params['solution_space'] is not None:
                # solution_space.set_params(self.params['solution_space'])

        # assigning instances created on attributes of the metaheuristic class
        self.Solution = solution_class
        self.neighborhood = neighborhood
        # self.solution_space = solution_space
        return self

    @abstractmethod
    def search(self) -> Solution:
        """ Performs metaheuristic search """

    def fit_search(self, problem: Problem) -> Solution:
        """ Fits and search """
        return self.fit(problem).search()

    def plot_evolution(self):
        # raise NotImplementedError
        pass
