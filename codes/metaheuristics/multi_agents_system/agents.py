from typing import List

from ..base_problem import Solution
from .sequencial_models import SequentialModel
from .base_agent import BaseAgent
from .q_learning import NeighborhoodQLearning
from .pools import BasePool


class QLearningAgent(NeighborhoodQLearning, BaseAgent):
    def __init__(self, unique_id: int, model: SequentialModel,
                 push_to: List[BasePool] = None, pull_from: List[BasePool] = None, reference_solution: Solution):
        BaseAgent.__init__(unique_id=unique_id, model=model, push_to=push_to, pull_from=pull_from)
        N = self.N
        NeighborhoodQLearning.__init__(neighborhood=N, )