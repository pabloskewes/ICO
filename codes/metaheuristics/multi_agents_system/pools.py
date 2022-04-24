from typing import List
from abc import ABC, abstractmethod

from ..base_problem import Solution, SolutionSpace


class BasePool(ABC):
    def __init__(self, solution_space: SolutionSpace, max_size: int = 10):
        self.SP = solution_space
        self.max_size = max_size
        self.solutions: List[Solution] = []

    @abstractmethod
    def push(self, solution: Solution):
        """  Tries to push a solution into the pool """


class DiversePool(BasePool):
    """ Pool of solution that keep only its space as diverse as possible """
    def __init__(self, solution_space: SolutionSpace, max_size: int = 10):
        super().__init__(solution_space, max_size)
        self.average_distances: List[float] = []

    def push(self, solution: Solution):
        if len(self.solutions) < self.max_size - 1:
            self.solutions.append(solution)
        elif len(self.solutions) == self.max_size - 1:
            self.solutions.append(solution)
            for sol_i in self.solutions:
                average_dist = sum((self.SP.distance(sol_i, sol_j) for sol_j in self.solutions
                                    if sol_i != sol_j)) / (self.max_size - 1)
                self.average_distances.append(average_dist)
        else:
            new_average_dist = sum((self.SP.distance(solution, sol_i) for sol_i in self.solutions)) \
                               / self.max_size
            max_average_dist = max(self.average_distances)
            if new_average_dist > max_average_dist:
                index = self.average_distances.index(max_average_dist)
                self.solutions[index] = solution
                self.average_distances[index] = new_average_dist

    def __repr__(self):
        display = ''
        display += f'pool max size = {self.max_size}'
        for sol, dist in zip(self.solutions, self.average_distances):
            display += f'Solution = {sol}'
            display += f'Average distance to pool = {dist}'
        return display
