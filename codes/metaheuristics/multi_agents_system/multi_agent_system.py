from __future__ import annotations
from typing import Dict, List, Union, Type, Optional, TYPE_CHECKING
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..base_metaheuristic import BaseMetaheuristic

if TYPE_CHECKING:
    from .sequential_models import SequentialModel, AgentCollection, PoolCollection
    from ..base_problem import Solution
    from .base_agent import BaseAgent
    from .pools import BasePool
    PoolCollectionClass = List[Union[str, Type[BasePool]]]


class MultiAgentSystem(BaseMetaheuristic):
    def __init__(self, model: Type[SequentialModel] = None, agents: AgentCollection = None,
                 pools: PoolCollectionClass = None, max_iter: int = 100, progress_bar: bool = False, verbose: int = 0,
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
        self.agents_collection = agents
        self.POOLS = pools
        self.pools: List[BasePool] = []
        self.verbose = verbose
        self.agents: List[BaseAgent] = []
        self.best_pool_solution: Optional[Solution] = None

        if None not in [self.POOLS, self.agents_collection, self.model]:
            self.setup()

    def setup(self):
        _, N, SP = self.get_problem_components()

        self.best_pool_solution = N.initial_solution()
        self.pools: List[BasePool] = [PoolClass(SP) for PoolClass in self.POOLS]

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
                if new_solution.cost() < self.best_pool_solution.cost():
                    self.best_solution = new_solution
        return self.best_solution

    def get_best_solution_from_agents(self) -> Solution:
        solutions = [agent.in_solution for agent in self.agents]
        costs = list(map(lambda s: s.cost(), solutions))
        index = costs.index(min(costs))
        return solutions[index]

    def search(self) -> Solution:
        self.run()
        pool_sol = None
        if any(hasattr(pool, 'get_best_solution') and pool.solutions for pool in self.pools):
            pool_sol = self.get_best_solution_from_pool()
        agent_sol = self.get_best_solution_from_agents()
        if pool_sol is None:
            new_sol = agent_sol
        else:
            new_sol = pool_sol if pool_sol.cost() <= agent_sol.cost() else agent_sol
        return new_sol

    def plot_agents_cost(self, ids=None, figsize=(14,7)):
        agents = self.get_agents()
        agents = agents if ids is None else [agent for i, agent in enumerate(agents) if i in ids]
        plt.figure(figsize=figsize)
        plt.title('Evolution of the cost of the found solutions')
        for agent in agents:
            plt.plot(agent.explored_solution_cost, label=f'solutions agent {agent.unique_id}')
        plt.xlabel('Time (iteration)')
        plt.ylabel('Cost of the solution')
        plt.legend()
        plt.show()

    def plot_agent_parallelism(self, height=0.4):
        epsilon = 0.01
        agents = self.get_agents()
        names = []
        for agent in agents:
            names.append(f'agent #{agent.unique_id}')
        plt.barh(names, width=self.max_iter, height=height, label='Agent iterations')
        for i, agent in enumerate(agents):
            for reset_step in agent.reset_steps:
                plt.plot([reset_step, reset_step], [i - (height / 2) + epsilon, i + (height / 2) - epsilon],
                         color='red')
        plt.plot([], [], color='red', label='Reset routine')
        plt.xlabel('Time (Iterations)')
        plt.ylabel('Agents')
        plt.title('Parallelism between agents')
        plt.legend()
