from __future__ import annotations
from typing import List, Any, Type, Tuple, Dict, TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from .sequencial_models import SequentialModel
    from .pools import BasePool
    from .pools import BasePool
    from .routines import BaseRoutine
    from ..base_problem import Solution


ReinforcedLearning = Any
Desires = Any


def empty_class():
    """ Empty class generator to avoid multiple inheritance of duplicate classes """
    class EmptyClass:
        def __init__(self, *args, **kwargs):
            pass
    return EmptyClass


def agent_factory(unique_id: int, model: SequentialModel, push_to: List[BasePool], pull_from: List[BasePool],
                  routine: Tuple[Type[BaseRoutine], Dict[str, str]] = None,
                  reinforced_learning: Tuple[Type[ReinforcedLearning], Dict[str, str]] = None,
                  desires: Tuple[Type[Desires], Dict[str, str]] = None,
                  initial_solution: Solution = None) -> BaseAgent:
    ROUTINE, routine_params = routine
    REINFORCED_LEARNING, rl_params = reinforced_learning
    DESIRES, desires_params = desires
    agent = BaseAgent(unique_id=unique_id, model=model, push_to=push_to, pull_from=pull_from)
    agent.set_init_solution(init_sol=initial_solution)
    agent.set_routine(ROUTINE)
    agent.set_reinforced_learning(REINFORCED_LEARNING)
    if agent.routine is not None:
        agent.routine.set_params(routine_params)
    if agent.rl is not None:
        agent.rl.set_params(rl_params)
    return agent


