from __future__ import annotations
from typing import Dict, List, Union, Type, Optional, TYPE_CHECKING
from tqdm import tqdm

from ..base_metaheuristic import BaseMetaheuristic

if TYPE_CHECKING:
    from .sequencial_models import SequentialModel
    from .pools import BasePool
    from .base_agent import BaseAgent
    from ..base_problem import Solution
    from .pools import BasePool

AgentCollection = Dict[Union[Type[BaseAgent], str], int]
PoolCollection = List[Union[str, BasePool]]


class MultiAgentSystem(BaseMetaheuristic):
    def __init__(self, model: Type[SequentialModel] = None, agents: AgentCollection = None,
                 pools: PoolCollection = None, max_iter: int = 100, progress_bar: bool = False,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()
        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

        # Metaheuristic hyperparameters
        self.max_iter = max_iter
        self.progress_bar = progress_bar
        self.MODEL = model
        self.model: Optional[SequentialModel] = None
        self.agents = agents
        self.pools = pools

        if self.pools is not None and self.agents is not None and self.model is not None:
            self.setup()

    def setup(self):
        self.model = self.MODEL(self.problem, self.agents, self.pools)

    def get_best_solution(self) -> Solution:
        for pool in self.pools:
            if hasattr(pool, 'get_best_solution'):
                new_solution = pool.get_best_solution()
                if new_solution.cost() < self.best_solution.cost():
                    self.best_solution = new_solution
        return self.best_solution

    def run(self):
        # Initialization of parameters
        self.setup()
        pbar = tqdm(total=self.max_iter) if self.progress_bar else None
        if self.progress_bar:
            pbar.set_description('Agents working...')

        for _ in range(self.max_iter):
            self.model.step()
            if self.progress_bar:
                pbar.update()

        if self.progress_bar:
            pbar.close()

    def search(self) -> Solution:
        self.run()
        return self.get_best_solution()
