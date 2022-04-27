from __future__ import annotations
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from abc import ABC, abstractmethod

from ..tabu_search import TabuList

if TYPE_CHECKING:
    from ..base_problem import Solution
    from .base_agent import BaseAgent


class Desire(ABC):
    """ aaa """
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    @abstractmethod
    def reward(self, solution: Solution):
        """ bbb """

    def set_params(self, params: Dict[str, Any]) -> None:
        """ Set parameters of routine """
        if params is None:
            return
        for varname, value in params.items():
            if not hasattr(self, varname):
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)
