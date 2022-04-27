from __future__ import annotations
from typing import List, Union, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..base_problem import Solution, SolutionSpace


class BasePool(ABC):
    """ Abstract pool of solutions for implementing different kind of pools """
    def __init__(self, solution_space: SolutionSpace, max_size: int = 10):
        self.SP = solution_space
        self.max_size = max_size
        self.solutions: List[Union[Solution, float]] = []
        self.n_push = 0
        self.n_pull = 0
        self.incoming_agent = None
        self.name = "BasePool"

    @abstractmethod
    def push(self, solution: Solution):
        """  Tries to push a solution into the pool """
        self.n_push += 1

    def receive_agent(self, agent):
        self.n_pull += 1
        self.incoming_agent = agent

    def display(self):
        """ Shows pool information """
        data = {'name': self.name, 'n_sols': len(self.solutions),
                'n_push': self.n_push, 'n_pull': self.n_pull}

        solution_string = ""
        for i in range(len(self.solutions)):
            solution_string += "S"+str(i)+" -> "+"%.2f"%self.solutions[i].cost()+"\n"

        final_string = ""
        final_string += "Pool {name}\n".format(**data)
        final_string += "------------------------------------------------------\n"
        final_string += "Number of solutions : {n_sols}\n".format(**data)
        final_string += "Number of 'pull' made : {n_pull}\n".format(**data)
        final_string += "Number of 'push' made : {n_push}\n".format(**data)
        # final_string += "Incoming Agent -> {agent_name}\n".format(**data)
        final_string += "Cost of solutions :\n"
        final_string += solution_string

        print(final_string, end='\r')


class BestPool(BasePool):
    """ Pool of solution that keeps the best solutions """
    def __init__(self, solution_space: SolutionSpace, max_size: int = 10):
        super().__init__(solution_space=solution_space, max_size=max_size)
        self.name = "BestPool"

    def push(self, solution: Solution):
        super().push(solution)
        if len(self.solutions) < self.max_size:
            self.solutions.append(solution)
        costs = [s.cost() for s in self.solutions]
        worst_sol = self.solutions[costs.index(max(costs))]
        if solution.cost() < worst_sol.cost():
            self.solutions.remove(worst_sol)
            self.solutions.append(solution)

    def get_best_solution(self) -> Solution:
        costs = [s.cost() for s in self.solutions]
        best_sol = self.solutions[costs.index(min(costs))]
        return best_sol


class BestScorePool(BasePool):
    """ Pool of solution that keeps only the best scores of solutions """
    def push(self, solution: Solution):
        super().push(solution)
        if len(self.solutions) < self.max_size:
            self.solutions.append(solution.cost())
        worst_sol = max(self.solutions)
        if solution.cost() < worst_sol:
            self.solutions.remove(worst_sol)
            self.solutions.append(solution.cost())


class DiversePool(BasePool):
    """ Pool of solution that keeps only its space as diverse as possible """
    def __init__(self, solution_space: SolutionSpace, max_size: int = 10):
        super().__init__(solution_space, max_size)
        self.average_distances: List[float] = []

    def push(self, solution: Solution):
        super().push(solution)
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

    def get_best_solution(self) -> Solution:
        costs = [s.cost() for s in self.solutions]
        best_sol = self.solutions[costs.index(min(costs))]
        return best_sol
