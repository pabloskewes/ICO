from typing import Optional, Dict
from mesa import Model, Agent
from mesa.time import RandomActivation

from .base_problem import Solution, Neighborhood, SolutionSpace, Problem
from .base_metaheuristic import BaseMetaheuristic
from .tabu_search import TabuSearch
from .simulated_annealing import SimulatedAnnealing


class CollaborativeAgents(BaseMetaheuristic):
    def __init__(self, agent_heuristics: Optional[Dict] = None, verbose: int = 0,
                 solution_params=None, neighborhood_params=None, solution_space_params=None):
        super().__init__()

        self.verbose = verbose
        self.agent_heuristics = agent_heuristics

        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

    def search(self) -> Solution:
        pass


class CollaborativeSpace(Model):
    def __init__(self, problem: Problem, agent_type: Agent, agent_heuristics):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.problem = problem
        self.agent_type = agent_type
        self.agent_heuristics = agent_heuristics
        unique_id = 0
        for agent_metaheuristic, agent_num in self.agent_heuristics.items():
            agent_metaheuristic = SimulatedAnnealing if agent_metaheuristic == 'simulated_annealing' else agent_metaheuristic
            agent_metaheuristic = TabuSearch if agent_metaheuristic == 'tabu_search' else agent_metaheuristic
            for i in range(agent_num):
                agent = agent_type(unique_id=unique_id, model=self, )
        # for i in range(self.num_agents):
        #     a = TabuAgent(i, self)
        #     self.schedule.add(a)
        #     # Add the agent to a random grid cell
        #     x = self.random.randrange(self.grid.width)
        #     y = self.random.randrange(self.grid.height)
        #     self.grid.place_agent(a, (x, y))

    def step(self):
        self.schedule.step()

