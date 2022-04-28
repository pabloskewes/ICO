from __future__ import annotations
from typing import List, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from IPython.display import display, clear_output

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
        self.max_iter = 0
        self.n_iter = 0
        self.reward_value = 0

    @abstractmethod
    def push(self, solution: Solution):
        """  Tries to push a solution into the pool """
        self.n_push += 1

    def receive_agent(self, agent):
        self.n_pull += 1
        self.incoming_agent = agent

    def set_iteration(self, n_iteration: int):
        self.n_iter = n_iteration

    def set_max_iteration(self, n_max: int):
        self.max_iter = n_max

    def display(self):
        """ Shows pool information """
        data = {'name': self.name, 'n_sols': len(self.solutions),
                'n_push': self.n_push, 'n_pull': self.n_pull,
                'n_iter': self.n_iter, 'max_iter': self.max_iter}
                # 'agent_name': 'Agent '+str(self.incoming_agent.unique_id)}

        clear_output(wait=True)
        display("Pool {name}".format(**data))
        display("------------------------------------------------------")
        display("Iteration {n_iter}/{max_iter}".format(**data))
        display("Number of solutions : {n_sols}".format(**data))
        display("Number of 'pull' made : {n_pull}".format(**data))
        display("Number of 'push' made : {n_push}".format(**data))
        # display("Incoming Agent -> {agent_name}".format(**data))
        display("Cost of solutions :")
        for i in range(len(self.solutions)):
            display("S"+str(i)+" -> "+"%.2f"%self.solutions[i].cost())


class BestPool(BasePool):
    """ Pool of solution that keeps the best solutions """
    def __init__(self, solution_space: SolutionSpace, max_size: int = 10):
        super().__init__(solution_space, max_size)
        self.name = "BestPool"

    def push(self, solution: Solution):
        super().push(solution=solution)
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
    def __init__(self, solution_space: SolutionSpace, max_size: int = 10):
        super().__init__(solution_space, max_size)
        self.name = "BestScorePool"

    def push(self, solution: Solution):
        super().push(solution=solution)
        if len(self.solutions) < self.max_size:
            self.solutions.append(solution.cost())
        worst_sol = max(self.solutions)
        if solution.cost() < worst_sol:
            self.solutions.remove(worst_sol)
            self.solutions.append(solution.cost())

    def set_reward_value(self, solution: Solution) -> None:
        rewards = []
        for cost in self.solutions:
            rewards.append(cost - solution.cost())
        try:
            self.reward_value = sum(rewards)/len(rewards)
        except ZeroDivisionError:
            self.reward_value = 0


class DiversePool(BasePool):
    """ Pool of solution that keeps only its space as diverse as possible """
    def __init__(self, solution_space: SolutionSpace, max_size: int = 10):
        super().__init__(solution_space, max_size)
        self.average_distances: List[float] = []
        self.name = "DiversePool"
        self.max_average_dist = 1

    def get_average_distance(self, solution: Solution = None) -> float:
        distance = sum((self.SP.distance(solution, sol_i) for sol_i in self.solutions))
        try:
            avg_dist = distance / len(self.solutions)
        except ZeroDivisionError:
            avg_dist = 0
        return avg_dist

    def push(self, solution: Solution):
        super().push(solution=solution)
        if len(self.solutions) < self.max_size - 1:
            self.solutions.append(solution)
        elif len(self.solutions) == self.max_size - 1:
            self.solutions.append(solution)
            for sol_i in self.solutions:
                average_dist = self.get_average_distance(sol_i)
                self.average_distances.append(average_dist)
        else:
            new_average_dist = self.get_average_distance(solution)
            self.max_average_dist = max(self.average_distances)
            if new_average_dist > self.max_average_dist:
                # print(f'len={len(self.average_distances)}')
                index = self.average_distances.index(self.max_average_dist)
                # print(index)
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

    def set_reward_value(self, solution: Solution) -> None:
        for sol_i in self.solutions:
            average_dist = self.get_average_distance(sol_i)
            self.average_distances.append(average_dist)
        new_average_dist = self.get_average_distance(solution)
        try:
            self.max_average_dist = max(self.average_distances)
        except ValueError:
            self.max_average_dist = 0
        self.reward_value = new_average_dist - self.max_average_dist
