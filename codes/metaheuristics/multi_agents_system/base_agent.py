from .sequencial_models import SequentialModel
from ..base_problem import Solution

from mesa import Agent as MesaAgent


class BaseAgent(MesaAgent):
    def __init__(self, unique_id: int, model: SequentialModel):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.problem = self.model.problem
        self.NEIGHBORHOOD = self.problem.NEIGHBORHOOD
        self.in_solution = None
        self.out_solution = None

    def step(self):
        pass

    def explore(self) -> Solution:
        """ Basic explore that simply looks for a neighborhood with the default configuration """
        N = self.NEIGHBORHOOD()
        self.out_solution = N(self.in_solution)
        return self.out_solution

