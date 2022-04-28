from __future__ import annotations
from typing import List, Optional, Any, Union, Type, Dict, TYPE_CHECKING
from mesa import Agent as MesaAgent
import random
import matplotlib.pyplot as plt

from ..base_problem import Solution
from .routines import Routine

if TYPE_CHECKING:
    from ..base_problem import Problem
    from .sequential_models import SequentialModel
    from .pools import BasePool
    from .q_learning import NeighborhoodQLearning
    from .desires import Desire
    ReinforcedLearning = Union[NeighborhoodQLearning, Any]


class BaseAgent(MesaAgent):
    def __init__(self, unique_id: int, model: SequentialModel, init_sol: Solution = None,
                 push_to: List[BasePool] = None, pull_from: List[BasePool] = None):
        super().__init__(unique_id, model)
        self.model: SequentialModel = model
        self.unique_id: int = unique_id
        self.problem: Problem = model.problem

        self.rl: ReinforcedLearning = None
        self.routine: Optional[Routine] = None
        self.desires: Optional[Desire] = None

        self.push_to: List[BasePool] = push_to
        self.pull_from: List[BasePool] = pull_from
        self.in_solution: Optional[Solution] = None
        self.ref_cost: Optional[float] = None
        self.set_init_solution(init_sol=init_sol)
        self.out_solution: Optional[Solution] = None

        self.N = self.problem.neighborhood()
        self.choose_pool: str = 'random'

        self.explored_solution_cost: List[float] = []
        self.reset_steps: List[int] = []

    def set_params(self, params: Dict[str, Any]) -> None:
        for varname, value in params.items():
            if not hasattr(self, varname):
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)

    def set_init_solution(self, init_sol: Optional[Solution]) -> None:
        if init_sol is None:
            init_sol = self.N.initial_solution()
        self.in_solution = init_sol
        self.ref_cost = init_sol.cost()

    def set_routine(self, ROUTINE: Type[Routine] = None) -> None:
        if ROUTINE is None:
            ROUTINE = Routine
        self.routine = ROUTINE(self)
        self.routine.reset_routine()

    def set_reinforced_learning(self, RL: Type[ReinforcedLearning]) -> None:
        if RL is not None:
            self.rl = RL(self)
            if self.desires:
                self.rl.is_desire = True

    def set_desires(self, DESIRES: Type[Desire]) -> None:
        if DESIRES is not None:
            self.desires = DESIRES(self)
            if self.rl:
                self.rl.is_desire = True

    def push_solution(self, solution: Solution) -> None:
        if self.desires:
            return self.desires.push_solution(solution)
        else:
            if self.push_to is None:
                return
            for pool in self.push_to:
                pool.push(solution)

    def pull_solution(self) -> Solution:
        """  Draws a solution from the pool """
        if self.desires:
            return self.desires.pull_solution()
        else:
            if not self.pull_from:
                return self.out_solution
            if self.choose_pool == 'random':
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
                raise Exception(f'{self.choose_pool} is not a valid form of choose_mode.')
            pool.receive_agent(self)
            return solution

    def reward(self, solution: Solution) -> float:
        """ The reward is defined by how much the solution found improves the reference solution. """
        if self.desires:
            return self.desires.reward(solution)
        else:
            return self.ref_cost - solution.cost()

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
            self.reset_steps.append(self.model.current_step)
            self.routine.reset_routine()
        self.in_solution = new_sol
        self.out_solution = None

    def plot_evolution_cost(self, figsize=(14, 7), resets=True):
        plt.figure(figsize=figsize)
        plt.title('Evolution of the cost of the found solutions')
        plt.plot(self.explored_solution_cost, c='orange', label='explored solution')
        line_color = 'red'
        if resets:
            for reset_step in self.reset_steps:
                plt.axvline(x=reset_step, color=line_color, linestyle='--')
            plt.plot([], [], color=line_color, label='Reset routine')
        plt.xlabel('Time (iteration)')
        plt.ylabel('Cost of the solution')
        plt.legend()
        plt.show()
