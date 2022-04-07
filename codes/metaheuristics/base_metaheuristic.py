from abc import ABC, abstractmethod
from typing import Tuple, Dict
from .base_problem import Solution, Neighborhood, SolutionSpace, Problem
import matplotlib.pyplot as plt


def _is_abstract(cls):
    """ Returns true if it is an abstract class """
    return bool(getattr(cls, "__abstractmethods__", False))


class BaseMetaheuristic(ABC):
    """
    Abstract class inherited from ABC (Abstract Base Class) class.
    It is used to create different algorithm classes.
    """
    def __init__(self):
        self.problem = None
        self.SOLUTION = None
        self.NEIGHBORHOOD = None
        self.SOLUTION_SPACE = None

        self.best_solution = None
        self.evolution_explored_solutions = []  # Liste des solutions explorés étudiées
        self.evolution_best_solution = []  # Liste des meilleures solutions (cout décroissante)

        self.params = {'solution': dict(), 'neighborhood': dict(), 'solution_space': dict()}
        self.static_params = {'solution': dict(), 'neighborhood': dict(), 'solution_space': dict()}

    def fit(self, problem: Problem):
        """ Fits a metaheuristic algorithm to a specific problem """
        self.problem = problem
        self.SOLUTION = problem.solution
        self.NEIGHBORHOOD = problem.neighborhood
        self.SOLUTION_SPACE = problem.solution_space
        return self

    @abstractmethod
    def search(self) -> Solution:
        """ Performs metaheuristic search """

    def fit_search(self, problem: Problem):
        """ Fits and search """
        return self.fit(problem).search()

    def get_problem_components(self) -> Tuple[Solution, Neighborhood, SolutionSpace]:
        """
        Instantiate a Solution, Neighborhood and SolutionSpace object in a tuple, with the parameters requested in
        the metaheuristic instantiation or simply with its default parameters.
        :return: Instances of Solution, Neighborhood and SolutionSpace object in a tuple, in that order.
        """
        if not _is_abstract(self.SOLUTION) and self.SOLUTION is not None:
            solution = self.SOLUTION(params=self.params['solution']) if self.params['solution'] is not None\
                else self.SOLUTION()
        else:
            solution = None
        if not _is_abstract(self.NEIGHBORHOOD) and self.NEIGHBORHOOD is not None:
            neighborhood = self.NEIGHBORHOOD(params=self.params['neighborhood']) if self.params['neighborhood'] is not None\
                else self.NEIGHBORHOOD()
        else:
            neighborhood = None
        if not _is_abstract(self.SOLUTION_SPACE) and self.SOLUTION_SPACE is not None:
            sol_space = self.SOLUTION_SPACE(params=self.params['solution_space']) if self.params['solution_space'] is not None\
                else self.SOLUTION_SPACE()
        else:
            sol_space = None
        return solution, neighborhood, sol_space

    def plot_evolution_cost(self):
        # plt.scatter(x=list(range(len(self.cost_list_best_sol))), y=self.cost_list_best_sol, c='turquoise')
        plt.title('Evolution of the cost of the found solutions')
        plt.plot(self.evolution_explored_solutions, c='turquoise', label='explored solutions')
        plt.plot(self.evolution_best_solution, c='orange', label='best solution')
        plt.xlabel('Time (iteration)')
        plt.ylabel('Cost of the solution')
        plt.legend()
        plt.show()
