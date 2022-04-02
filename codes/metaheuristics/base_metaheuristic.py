from abc import ABC, abstractmethod
from problem import Problem


class BaseMetaheuristic(ABC):
    """
    Abstract class inherited from ABC (Abstract Base Class) class.
    It is used to create different algorithm classes.
    """
    def __init__(self):
        self.best_solution = None
        self.solution = None
        self.neighborhood = None
        self.cost_list = []
        self.params = None

    def fit(self, problem: Problem):
        """ Fits a metaheuristic algorithm to a specific problem """
        solution = problem.solution
        solution.set_params(self.params['solution'])
        self.solution = problem.solution
        self.neighborhood = problem.neighborhood
        return self

    @abstractmethod
    def search(self):
        """ Performs metaheuristic search """

    def fit_search(self, problem: Problem):
        """ Fits and search """
        return self.fit(problem).search()

    def plot_evolution(self):
        raise NotImplementedError
