from .base_agent import BaseAgent
from .pools import BasePool
from ..base_problem import Problem

from mesa import Model as MesaModel
from mesa.time import RandomActivation
from typing import List, Dict, Union, Type

AgentCollection = Dict[Union[Type[BaseAgent], str], int]
PoolCollection = List[Union[str, BasePool]]


class SequentialModel(MesaModel):
    def __init__(self, problem: Problem, agents: AgentCollection, push_to: BasePool = None, pull_from: BasePool = None):
        super().__init__()
        self.problem = problem
        self.agents_types = agents
        self.push_to = push_to
        self.pull_from = pull_from
        self.schedule = RandomActivation(self)

        self.set_agents_types()
        self.init_agents()

    def set_agents_types(self):
        """ Transform string coded agents in "agent_types" to agent classes """
        # TODO: Update this whenever we have agents
        pass

    def init_agents(self):
        unique_id = 0
        for agent_class, num in self.agents_types:
            for i in range(num):
                agent = agent_class(unique_id=unique_id, model=self, push_to=self.push_to, pull_from=self.pull_from)
                self.schedule.add(agent)
                unique_id += 1

    def step(self):
        self.schedule.step()
