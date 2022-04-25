from __future__ import annotations
from typing import Dict, List, Union, Type, Optional, TYPE_CHECKING
from tqdm import tqdm

from ..base_metaheuristic import BaseMetaheuristic

if TYPE_CHECKING:
    from .sequential_models import SequentialModel, AgentCollection, PoolCollection
    from ..base_problem import Solution
    from .base_agent import BaseAgent


class MultiAgentSystem(BaseMetaheuristic):
    def __init__(self, model: Type[SequentialModel] = None, agents: AgentCollection = None,
                 pools: PoolCollection = None, max_iter: int = 100, progress_bar: bool = False, verbose: int = 0,
                 solution_params=None, neighborhood_params=None, solution_space_params=None, Li=List):
        super().__init__()
        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

        # Metaheuristic hyperparameters
        self.max_iter = max_iter
        self.progress_bar = progress_bar
        self.MODEL = model
        self.model: Optional[SequentialModel] = None
        self.agents_collection = agents
        self.pools = pools
        self.verbose = verbose
        self.agents: List[BaseAgent] = []

        if None not in [self.pools, self.agents_collection, self.model]:
            self.setup()

    def setup(self):
        self.model = self.MODEL(problem=self.problem, agents=self.agents_collection,
                                push_to=self.pools, pull_from=self.pools,
                                verbose=self.verbose)
        self.agents = self.get_agents()

    def get_agents(self) -> List[BaseAgent]:
        return self.model.schedule.agents

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

    def get_best_solution_from_pool(self) -> Solution:
        for pool in self.pools:
            if hasattr(pool, 'get_best_solution'):
                new_solution = pool.get_best_solution()
                if new_solution.cost() < self.best_solution.cost():
                    self.best_solution = new_solution
        return self.best_solution

    def get_best_solution_from_agents(self) -> Solution:
        solutions = [agent.in_solution for agent in self.agents]
        costs = list(map(lambda s: s.cost(), solutions))
        best_cost = min(costs)
        index = costs.index(best_cost)
        return solutions[index]

    def search(self) -> Solution:
        self.run()
        if self.pools:
            return self.get_best_solution_from_pool()
        else:
            return self.get_best_solution_from_agents()
