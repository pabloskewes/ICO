from mesa import Agent as MesaAgent
from typing import List, Optional
import random

from .sequencial_models import SequentialModel
from ..base_problem import Solution
from .pools import BasePool
from .routines import BaseRoutine


class BaseAgent(MesaAgent):
    def __init__(self, unique_id: int, model: SequentialModel,
                 push_to: List[BasePool] = None, pull_from: List[BasePool] = None):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.problem = model.problem
        self.N = self.problem.neighborhood()
        self.push_to = push_to
        self.pull_from = pull_from
        self.in_solution = None
        self.out_solution = None
        self.routine: Optional[BaseRoutine] = None
        self.explored_solution_cost: List[float] = []

    def set_init_solution(self, init_sol: Optional[Solution]):
        if init_sol is None:
            init_sol = self.N.initial_solution()
        self.in_solution = init_sol

    def set_routine(self, routine: BaseRoutine):
        self.routine = routine

    def push_solution(self, solution: Solution) -> None:
        if self.push_to is None:
            return
        for pool in self.push_to:
            pool.push(solution)

    def pull_solution(self, choose_mode='random') -> Solution:
        """  Draws a solution from the pool """
        if not self.pull_from:
            return self.out_solution
        if choose_mode == 'random':
            pool = random.choice(self.pull_from)
            breaker = 10
            while not pool and breaker:
                pool = random.choice(self.pull_from)
                breaker -= 1
            if pool:
                r_index = random.choice(range(len(pool.solutions)))
                solution = pool.solutions.pop(r_index)
            else:
                solution = self.out_solution
        else:
            raise Exception(f'{choose_mode} is not a valid form of choose_mode.')
        return solution

    def explore(self, solution: Solution) -> Solution:
        """ Basic explore that simply looks for a neighborhood with the default configuration. It's normally overridden"""
        N = self.N
        new_sol = N(solution)
        return new_sol

    def step(self):
        if not self.routine.is_finished:
            new_sol = self.routine.iteration()
            self.explored_solution_cost.append(new_sol.cost())
            self.push_solution(new_sol)
        else:
            new_sol = self.pull_solution()
        self.in_solution = new_sol
        self.out_solution = None



