from .sequencial_models import SequentialModel
from ..base_problem import Solution

from mesa import Agent as MesaAgent


class BaseAgent(MesaAgent):
    def __init__(self, unique_id: int, model: SequentialModel):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.problem = model.problem
        self.pools = model.pools
        self.N = self.problem.NEIGHBORHOOD()
        self.in_solution = None
        self.out_solution = None

    def explore(self, solution: Solution) -> Solution:
        """ Basic explore that simply looks for a neighborhood with the default configuration. It's normally overridden"""
        N = self.N
        new_sol = N(solution)
        return new_sol

