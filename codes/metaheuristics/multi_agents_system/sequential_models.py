from __future__ import annotations
from typing import List, Dict, Union, Type, Tuple, TYPE_CHECKING
from mesa import Model as MesaModel
from mesa.time import RandomActivation

from .agents import assemble_agent

if TYPE_CHECKING:
    from .pools import BasePool
    from ..base_problem import Problem
    from .agents import AgentStructure

    AgentCollection = Dict[Union[AgentStructure, str], int]
    PoolCollection = List[Union[str, BasePool]]


class SequentialModel(MesaModel):
    def __init__(self, problem: Problem, agents: AgentCollection,
                 push_to: BasePool = None, pull_from: BasePool = None,
                 verbose: int = 0):
        super().__init__()
        self.problem = problem
        self.agents_types = agents
        self.push_to = push_to
        self.pull_from = pull_from
        self.schedule = RandomActivation(self)
        self.verbose = verbose
        self.step = 0

        self.set_agents_types()
        self.init_agents()

    def set_agents_types(self):
        """ Transform string coded agents in "agent_types" to agent classes """
        # TODO: Update this whenever we have agents
        pass

    def init_agents(self):
        N = self.problem.neighborhood()
        init_sol = N.initial_solution()
        unique_id = 0
        for agent_structure, num in self.agents_types.items():
            for i in range(num):
                if self.verbose >= 2:
                    print(f'Assembling agent #{unique_id}')
                agent = assemble_agent(unique_id=unique_id, model=self, push_to=self.push_to, pull_from=self.pull_from,
                                       agent_structure=agent_structure, initial_solution=init_sol)
                self.schedule.add(agent)
                unique_id += 1

        if self.verbose >= 1:
            print('All agents were successfully created')

    def step(self):
        self.schedule.step()
        self.step += 1
