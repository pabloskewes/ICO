from __future__ import annotations
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from abc import ABC, abstractmethod
import random

from .pools import BestScorePool, DiversePool

if TYPE_CHECKING:
    from ..base_problem import Solution
    from .base_agent import BaseAgent


class Desire(ABC):
    """ aaa """
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    @abstractmethod
    def reward(self, solution: Solution) -> float:
        """ bbb """

    @abstractmethod
    def push_solution(self, solution: Solution) -> None:
        """ bbb """

    @abstractmethod
    def pull_solution(self) -> Solution:
        """ bbb """

    def set_params(self, params: Dict[str, Any]) -> None:
        """ Set parameters of routine """
        if params is None:
            return
        for varname, value in params.items():
            if not hasattr(self, varname):
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)


class Competition(Desire):

    def __init__(self, agent: BaseAgent):
        super().__init__(agent)
        self.pool = None
        self.set_pool()

    def set_pool(self):
        self.pool = random.choice(self.agent.pull_from)
        breaker = 10
        while not isinstance(self.pool, BestScorePool) and breaker:
            self.pool = random.choice(self.agent.pull_from)
            breaker -= 1
        if not isinstance(self.pool, BestScorePool):
            self.pool = None

    def pull_solution(self) -> Solution:
        if self.pool is None:
            return self.agent.out_solution
        else:
            r_index = random.choice(range(len(self.pool.solutions)))
            solution = self.pool.solutions.pop(r_index)
        return solution

    def push_solution(self, solution: Solution) -> None:
        if self.agent.push_to is None:
            return
        for pool in self.agent.push_to:
            pool.push(solution)

    def reward(self, solution: Solution) -> float:
        original_reward = self.agent.rl.ref_cost - solution.cost()
        desire_extra_value = 0
        if self.pool:
            reward_value = self.pool.reward_value
            try:
                desire_extra_value = (reward_value/max(self.pool.solutions))*original_reward \
                                     + min(self.agent.rl.ref_cost, solution.cost())
            except ValueError:
                desire_extra_value = 0
        return desire_extra_value + original_reward


class Diversification(Desire):
    def __init__(self, agent: BaseAgent):
        super().__init__(agent)
        self.pool = None
        self.set_pool()

    def set_pool(self):
        self.pool = random.choice(self.agent.pull_from)
        breaker = 10
        while not isinstance(self.pool, DiversePool) and breaker:
            self.pool = random.choice(self.agent.pull_from)
            breaker -= 1
        if not isinstance(self.pool, DiversePool):
            self.pool = None

    def pull_solution(self) -> Solution:
        if self.pool is None:
            return self.agent.out_solution
        else:
            r_index = random.choice(range(len(self.pool.solutions)))
            solution = self.pool.solutions.pop(r_index)
        return solution

    def push_solution(self, solution: Solution) -> None:
        if self.agent.push_to is None:
            return
        for pool in self.agent.push_to:
            pool.push(solution)

    def reward(self, solution: Solution) -> float:
        original_reward = self.agent.ref_cost - solution.cost()
        desire_extra_value = 0
        if self.pool:
            reward_value = self.pool.reward_value
            try:
                desire_extra_value = (reward_value/self.pool.max_average_dist)*original_reward \
                                        + min(self.agent.rl.ref_cost, solution.cost())
            except ZeroDivisionError:
                desire_extra_value = 0

        return desire_extra_value + original_reward
