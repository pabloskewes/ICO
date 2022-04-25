from __future__ import annotations
from typing import List, Optional, Any, Union, Type, TYPE_CHECKING
from mesa import Agent as MesaAgent
import random

from ..base_problem import Solution
from .routines import BaseRoutine

if TYPE_CHECKING:
    from .sequencial_models import SequentialModel
    from .pools import BasePool
    from .q_learning import NeighborhoodQLearning


ReinforcedLearning = Union[NeighborhoodQLearning, Any]


class BaseAgent(MesaAgent):
    def __init__(self, unique_id: int, model: SequentialModel,
                 push_to: List[BasePool] = None, pull_from: List[BasePool] = None):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.problem = model.problem
        self.N = self.problem.neighborhood()
        self.rl: ReinforcedLearning = None
        self.push_to = push_to
        self.pull_from = pull_from
        self.in_solution = None
        self.out_solution = None
        self.routine: Optional[BaseRoutine] = None
        self.explored_solution_cost: List[float] = []

    def set_init_solution(self, init_sol: Optional[Solution]) -> None:
        if init_sol is None:
            init_sol = self.N.initial_solution()
        self.in_solution = init_sol

    def set_routine(self, ROUTINE: Type[BaseRoutine] = None) -> None:
        if ROUTINE is None:
            ROUTINE = BaseRoutine
        self.routine = ROUTINE(self)

    def set_reinforced_learning(self, RL: Type[ReinforcedLearning]) -> None:
        if RL is not None:
            self.rl = RL(self)

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
        """  """
        if self.rl:
            new_sol = self.rl.explore(solution)
        else:
            new_sol = self.N(solution)
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



