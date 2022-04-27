from __future__ import annotations
from typing import List, Any, Type, Tuple, Dict, Union, Optional, TYPE_CHECKING
from mesa import Model as MesaModel

from .base_agent import BaseAgent
from .routines import Routine

if TYPE_CHECKING:
    from .sequential_models import SequentialModel
    from .pools import BasePool
    from .pools import BasePool
    from ..base_problem import Solution, Problem

    ReinforcedLearning = Any
    Desires = Any
    FeaturesParams = Dict[str, Optional[Dict[str, Any]]]
    # AgentFeatures = Dict[str, Union[Type[Routine], Type[ReinforcedLearning], Type[Desires]]]
    # RawAgentFeatures = Union[AgentFeatures, Tuple[Type[Routine], Type[ReinforcedLearning], Type[Desires]]]
    # RawAgentMold = Union[AgentFeatures, Tuple[AgentFeatures, FeaturesParams], Dict[str, Union[AgentFeatures, FeaturesParams]]]


class AgentStructure:
    ROUTINE: Type[Routine] = None
    REINFORCED_LEARNING: Type[ReinforcedLearning] = None
    DESIRES: Type[Desires] = None
    features_params: FeaturesParams = None

    @classmethod
    def print_pieces(cls):
        print('Showing pieces of Agent Structure')
        print('Routine =', cls.ROUTINE)
        print('RL =', cls.REINFORCED_LEARNING)
        print('Desires =', cls.DESIRES)


class AgentFactory:
    def __init__(self, routine: Routine = None,
                 reinforced_learning: ReinforcedLearning = None,
                 desires: Desires = None, features_params: FeaturesParams = None):
        self.ROUTINE = routine if routine is not None else Routine
        self.REINFORCED_LEARNING = reinforced_learning
        self.DESIRES = desires
        self.features_params = features_params

    def build(self) -> Type[AgentStructure]:
        params = self.features_params if self.features_params is not None else {}
        keys = ['routine', 'reinforced_learning', 'desires', 'agent_params']
        for key in keys:
            if key not in params:
                params[key] = None

        class NewMold(AgentStructure):
            ROUTINE = self.ROUTINE
            REINFORCED_LEARNING = self.REINFORCED_LEARNING
            DESIRES = self.DESIRES
            features_params = params
        return NewMold

    def __call__(self, routine: Routine = None,
                 reinforced_learning: ReinforcedLearning = None,
                 desires: Desires = None, features_params: FeaturesParams = None) -> Type[AgentStructure]:
        self.ROUTINE = routine if routine is not None else Routine
        self.REINFORCED_LEARNING = reinforced_learning
        self.DESIRES = desires
        self.features_params = features_params
        return self.build()


def assemble_agent(unique_id: int, model: SequentialModel, agent_structure: AgentStructure,
                   push_to: List[BasePool] = None, pull_from: List[BasePool] = None,
                   initial_solution: Solution = None) -> BaseAgent:
    agent = BaseAgent(unique_id=unique_id, model=model, init_sol=initial_solution, push_to=push_to, pull_from=pull_from)
    if agent_structure.ROUTINE is not None:
        agent.set_routine(agent_structure.ROUTINE)
        agent.routine.set_params(agent_structure.features_params['routine'])
    if agent_structure.REINFORCED_LEARNING is not None:
        agent.set_reinforced_learning(agent_structure.REINFORCED_LEARNING)
        agent.rl.set_params(agent_structure.features_params['reinforced_learning'])
    # TODO: Include Desires setting
    if agent_structure.features_params['agent_params'] is not None:
        agent.set_params(agent_structure.features_params['agent_params'])

    return agent


# def static_agent(problem: Problem, agent_structure: AgentStructure, push_to: List[BasePool] = None,
#                  pull_from: List[BasePool] = None, initial_solution: Solution = None):
#     model = SequentialModel(problem=problem, agents={agent_structure: 1}, push_to=push_to, pull_from=pull_from)
#     agent = assemble_agent(unique_id=100, model=model, agent_structure=agent_structure, push_to=push_to,
#                            pull_from=pull_from, initial_solution=initial_solution)
#     return model.schedule.agents[0]

# ==================================================================================================================
# ==================================================================================================================
# ====================================== SAVING FOR LATER IN CASE OF ===============================================
# ==================================================================================================================
# ==================================================================================================================


# def put_tuple_into_dict(dictionary: Dict[str, Any], tup, names: List[str]) -> None:
#     """ Puts info of a tuple in a dictionary according to a given list of names """
#     assert len(tup) == len(names), 'tuple must be of same size as names'
#     for x, name in zip(list(tup), names):
#         if x is None:
#             continue
#         dictionary[name] = x
#
#
# def prepare_agent_features(raw_agent_features: RawAgentFeatures) -> AgentFeatures:
#     agent_features_dict = {'routine': None, 'reinforced_learning': None, 'desires': None}
#     if raw_agent_features is None:
#         return agent_features_dict
#     if type(raw_agent_features) == tuple:
#         # we assume type(raw_agent_features) == Tuple[Type[Routine], Type[ReinforcedLearning], Type[Desires]]
#         put_tuple_into_dict(agent_features_dict, raw_agent_features, ['routine', 'reinforced_learning', 'desires'])
#     elif type(raw_agent_features) == dict:
#         # we assume type(raw_agent_features) == AgentFeatures
#         if 'routine' in raw_agent_features:
#             agent_features_dict['routine'] = raw_agent_features['routine']
#         if 'reinforced_learning' in raw_agent_features:
#             agent_features_dict['reinforced_learning'] = raw_agent_features['reinforced_learning']
#         if 'desires' in raw_agent_features:
#             agent_features_dict['desires'] = raw_agent_features['desires']
#     else:
#         raise Exception(f"raw_agent_features can't take value {raw_agent_features}. It must be AgentFeatures or "
#                         f"Tuple[Routine, ReinforcedLearning, Desires]")
#     return agent_features_dict
#
#
# def prepare_agent_mold(raw_agent_mold: RawAgentMold):
#     agent_mold_dict = {'agent_features': None, 'features_params': None}
#     if type(raw_agent_mold) == tuple:
#         if len(raw_agent_mold) == 3:
#             # we assume type(raw_agent_mold) == AgentFeatures
#             agent_mold_dict['agent_features'] = raw_agent_mold
#         elif len(raw_agent_mold) == 2 and type(raw_agent_mold[0]) == tuple:
#             # we assume type(raw_agent_mold) == Tuple[AgentFeatures, FeaturesParams]
#             put_tuple_into_dict(agent_mold_dict, raw_agent_mold, ['agent_features', 'features_params'])
#     elif type(raw_agent_mold) == dict:
#         agent_mold_dict['agent_features'] = raw_agent_mold['agent_features']
#         if 'features_params' in raw_agent_mold:
#             agent_mold_dict['features_params'] = raw_agent_mold['features_params']
#     else:
#         raise Exception(f"raw_agent_mold can't take value {raw_agent_mold}. It must be AgentFeatures or "
#                         f"Tuple[AgentFeatures, FeaturesParams] or Dict[str, Union[AgentFeatures, FeaturesParas]. "
#                         f"If a dictionary is given, it must have the 'agent_features', and it can have 'feature_params'")
#     agent_mold_dict['agent_features'] = prepare_agent_features(agent_mold_dict['agent_features'])
#     return agent_mold_dict
#
#
# def agent_factory(unique_id: int, model: SequentialModel, agent_mold: AgentMold,
#                   push_to: List[BasePool] = None, pull_from: List[BasePool] = None,
#                   initial_solution: Solution = None) -> BaseAgent:
#     agent_mold_dict = prepare_agent_mold(agent_mold)
#     agent_features, features_params = agent_mold_dict['agent_features'], agent_mold_dict['features_params']
#
#     if agent_features is None:
#         agent_features = {'routine': None, 'reinforced_learning': None, 'desires': None}
#     if features_params is None:
#         features_params = {'routine': None, 'reinforced_learning': None, 'desires': None}
#     ROUTINE, REINFORCED_LEARNING, DESIRES = \
#         agent_features['routine'], agent_features['reinforced_learning'], agent_features['desires']
#     routine_params, rl_params, desires_params = \
#         features_params['routine'], features_params['reinforced_learning'], features_params['desires']
#
#     agent = BaseAgent(unique_id=unique_id, model=model, init_sol=initial_solution, push_to=push_to, pull_from=pull_from)
#
#     if ROUTINE is not None:
#         agent.set_routine(ROUTINE)
#         agent.routine.set_params(routine_params)
#     if REINFORCED_LEARNING is not None:
#         agent.set_reinforced_learning(REINFORCED_LEARNING)
#         agent.rl.set_params(rl_params)
#     # TODO: include setting of desires params
#     return agent


