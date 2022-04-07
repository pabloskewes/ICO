from mesa import Agent
from mesa import Model as MesaModel
from mesa.time import RandomActivation
from typing import Optional, Dict, Any, List, Type, Union

from .base_metaheuristic import BaseMetaheuristic
from .simulated_annealing import SimulatedAnnealing
from .tabu_search import TabuSearch
from .base_problem import Problem, Solution


class SimpleAgent(Agent):
    def __init__(self, unique_id, model, metaheuristic: Type[BaseMetaheuristic],
                 solution_pool, in_solution: Solution = None):
        super().__init__(unique_id, model)

        self.metaheuristic = metaheuristic()
        self.problem = self.model.problem
        self.solution_pool = solution_pool
        self.metaheuristic.fit(self.problem)
        self.in_solution = in_solution
        self.out_solution = None

    def step(self):
        self.out_solution = self.metaheuristic.search()
        self.solution_pool.push(self.out_solution)

    def get_out_solution(self) -> Solution:
        return self.out_solution


AgentType = SimpleAgent
MetaheuristicDict = Dict[Union[str, BaseMetaheuristic], int]


class CollaborativeSpace(MesaModel):
    def __init__(self, problem: Problem, agent_type: Type[AgentType], agent_heuristics: MetaheuristicDict,
                 solution_pool):
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
                agent = agent_type(unique_id=unique_id, model=self, metaheuristic=agent_metaheuristic,
                                   solution_pool=solution_pool)
                self.schedule.add(agent)
                unique_id += 1

    def step(self):
        self.schedule.step()
