from mesa import Agent
from typing import Optional, Dict

from .base_metaheuristic import BaseMetaheuristic
from .base_problem import Problem


class SimpleAgent(Agent):
    def __init__(self, unique_id, model, metaheuristic: BaseMetaheuristic, problem: Problem,
                 params: Optional[Dict] = None):
        super().__init__(unique_id, model)

        self.metaheuristic = metaheuristic(params)
        self.tabu_model.fit(model.problem)

        self.in_solution = None
        self.out_solution = None

    def step(self):
        self.solution = self.tabu_model.search()