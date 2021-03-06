from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Type
import inspect


class Context:
    """
    The context represents the data that will instantiate a problem. e.g., in a machine task ordering problem,
    the context would represent the data that indicates how long it takes each machine to do its task, what tasks must
    be done before each task can begin, and so on.
    """


class ProblemComponent:
    """
    A problem component is an abstract class that inherits Solution, Neighborhood and SolutionSpace, which contains
    general methods for all these components. The implementation of these child classes represents the modeling of the
    problem component.
    """
    static_valid_params = []

    def __init__(self, params=None):
        self.params = params
        self.valid_params = []

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Sets attributes to a problem component instance (only if it is in the list of attributes that are allowed
        to edit "valid_params").
        :param params: Dictionary of attributes to be set
        :return: None
        """
        for varname, value in params.items():
            if varname not in self.valid_params:
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)

    @classmethod
    def set_class_context(cls, context):
        """ Sets a static attribute to the class, containing the context """
        setattr(cls, 'context', context)

    @classmethod
    def set_class_params(cls, params):
        """
        Sets static attributes to a problem component class (only if it is in the list of attributes that are allowed
        to edit "static_valid_params").
        :param params: Dictionary of attributes to be set
        :return: None
        """
        for varname, value in params.items():
            if varname not in cls.static_valid_params:
                raise Exception(f'{varname} is not a valid static keyword argument')
            setattr(cls, varname, value)

    @classmethod
    def get_class_params(cls) -> Dict[str, Any]:
        """ Retrieves the static attributes of the class, excluding methods, context and static_valid_params list. """
        params = {}
        for param in inspect.getmembers(cls):
            if not param[0].startswith('_') and not inspect.ismethod(param[1]) and\
                    not inspect.isfunction(param[1]) and param[0] not in ['static_valid_params', 'context']:
                param_name = param[0]
                params[param_name] = getattr(cls, param_name)
        return params


class Solution(ABC, ProblemComponent):
    """ Class where the "shape" of the solutions is modeled. """
    @abstractmethod
    def cost(self) -> float:
        """ The cost of a solution is defined """


class Neighborhood(ABC, ProblemComponent):
    """ Class where the methods to explore the neighborhoods of the solutions are designed. """
    def __init__(self):
        super(ProblemComponent).__init__()
        self.use_methods: List[Any] = []

    @abstractmethod
    def initial_solution(self) -> Solution:
        """ Defines an initial solution for use in the metaheuristic process """

    @abstractmethod
    def get_neighbor(self, solution: Solution) -> Solution:
        """ The way in which the neighborhoods of a solution are chosen is defined. """

    def __call__(self, solution: Solution):
        """ Method to be able to search a neighborhood with the syntax new_s = N(s) """
        return self.get_neighbor(solution)


class SolutionSpace(ABC, ProblemComponent):
    """ Class where the solution space is modeled: the relationships between solutions and where tools such as "Pools"
    can be designed to store solutions."""
    def __init__(self):
        super().__init__()
        self.diverse_pool = None

    @abstractmethod
    def distance(self, s1: Solution, s2: Solution) -> float:
        """ Defines the distance between 2 solutions in the solution space. """


class Problem:
    """ Class that models a problem. It packages the other implemented classes into a single class. """
    solution: Type[Solution] = None
    neighborhood: Type[Neighborhood] = None
    solution_space: Type[SolutionSpace] = None

    def __init__(self, context: Context):
        self.context = context

        self.default_solution_params = self.solution.get_class_params() if self.solution is not None\
            else None
        self.default_neighborhood_params = self.neighborhood.get_class_params() if self.neighborhood is not None\
            else None
        self.default_solution_space_params = self.solution_space.get_class_params() if self.solution_space is not None\
            else None

        if hasattr(self, 'solution') and self.solution is not None:
            self.solution.set_class_context(context)
        if hasattr(self, 'neighborhood') and self.neighborhood is not None:
            self.neighborhood.set_class_context(context)
        if hasattr(self, 'solution_space') and self.solution_space is not None:
            self.solution_space.set_class_context(context)

    @classmethod
    def set_problem_params(cls, solution_params: Optional[Dict[str, Any]] = None,
                           neighborhood_params: Optional[Dict[str, Any]] = None,
                           solution_space_params: Optional[Dict[str, Any]] = None):
        """ Sets static attributes to the problem components (Solution, Neighborhood and Solution Space). """
        if solution_params is not None:
            cls.solution.set_class_params(solution_params)
        if neighborhood_params is not None:
            cls.neighborhood.set_class_params(neighborhood_params)
        if solution_space_params is not None:
            cls.neighborhood.set_class_params(solution_space_params)

    def reset_class_params(self):
        """
        Resets the static attributes of the problem components to their default attributes, which were saved the
        first time the first problem of that type was instantiated.
        """
        if self.solution is not None:
            self.solution.set_class_params(self.default_solution_params)
        if self.neighborhood is not None:
            self.neighborhood.set_class_params(self.default_neighborhood_params)
        if self.solution_space is not None:
            self.solution_space.set_class_params(self.default_solution_space_params)

    def print_class_params(self):
        """ Prints static attributes of the problem components. """
        print('Static parameters in problem components:')
        if self.solution is not None:
            print('Solution class :', self.solution.get_class_params())
        if self.neighborhood is not None:
            print('Neighborhood class :', self.neighborhood.get_class_params())
        if self.solution_space is not None:
            print('Solution Space class: ', self.solution_space.get_class_params())

    def _print_default_class_params(self):
        """ Prints default static attributes of the problem components. """
        print('Default static parameters in problem components:')
        if self.default_solution_params is not None:
            print('Solution class :', self.default_solution_params)
        if self.default_neighborhood_params is not None:
            print('Neighborhood class :', self.default_neighborhood_params)
        if self.default_solution_space_params is not None:
            print('Solution Space class: ', self.default_solution_space_params)

    @classmethod
    def set_static_params(cls, solution_params: Optional[Dict[str, Any]] = None,
                          neighborhood_params: Optional[Dict[str, Any]] = None,
                          solution_space_params: Optional[Dict[str, Any]] = None):
        """ Alias for set_problem_params class method. """
        cls.set_problem_params(solution_params, neighborhood_params, solution_space_params)

    def print_static_params(self):
        """ Alias for print_class_params method. """
        self.print_class_params()

