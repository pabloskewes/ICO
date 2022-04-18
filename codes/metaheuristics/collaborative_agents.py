from typing import Optional, Dict, Callable, Any, Union, Type
from mesa import Model, Agent
from mesa.time import RandomActivation
from tqdm import tqdm

from .base_problem import Solution, Neighborhood, SolutionSpace, Problem
from .base_metaheuristic import BaseMetaheuristic
from .tabu_search import TabuSearch
from .simulated_annealing import SimulatedAnnealing
from .agents import CollaborativeSpace, SimpleAgent
AgentType = SimpleAgent


class CollaborativeAgents(BaseMetaheuristic):
    def __init__(self, agent_type: str = 'simple_agent', agent_heuristics: Optional[Dict] = None,
                 pool_size: int = 10, steps: int = 50, final_search: str = 'tabu_search', verbose: int = 0,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.verbose = verbose
        self.agent_heuristics = agent_heuristics
        self.pool_size = pool_size
        self.agent_type = SimpleAgent if agent_type == 'simple_agent' else None
        self.steps = steps
        self.final_search = final_search

        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

    def search(self) -> Solution:
        _, _, solution_space = self.get_problem_components()
        solution_space.set_diverse_pool(size=self.pool_size)
        solution_pool = solution_space.diverse_pool
        collaborative_space = CollaborativeSpace(problem=self.problem, agent_type=self.agent_type,
                                                 agent_heuristics=self.agent_heuristics, solution_pool=solution_pool)
        for _ in tqdm(range(self.steps)):
            collaborative_space.step()

        if self.verbose >= 1:
            print(solution_pool)

        final_heuristic = TabuSearch if self.final_search == 'tabu_search' else None
        final_heuristic = SimulatedAnnealing if self.final_search == 'simulated_annealing' else None
        assert final_heuristic is not None, f'{final_heuristic} is not a valid heuristic model for final search'

        final_candidates = []
        cost_candidates = []
        for solution in solution_pool.solutions:
            final_candidate = final_heuristic(neighborhood_params={'init_sol': solution}).fit_search(self.problem)
            final_candidates.append(final_candidate)
            cost_candidates.append(final_candidate.cost())
        best_index = cost_candidates.index(min(cost_candidates))
        best_solution = final_candidates[best_index]

        self.best_solution = best_solution
        return best_solution








