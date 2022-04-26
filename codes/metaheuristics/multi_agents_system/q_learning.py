from __future__ import annotations
from typing import List, Any, Dict, Optional, TYPE_CHECKING
import numpy as np
from pandas import DataFrame
import random
from abc import ABC, abstractmethod

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from ..base_problem import Solution, Neighborhood

State = Any
Action = Any


class QLearning(ABC):
    def __init__(self, states: List[Any], actions: List[Any]):
        """
        Initializes the main components of a Q-learning:
        - states: States in which an agent can be found
        - actions: Actions that an agent can perform, and lead to another state.
        - alpha: Learning Rate
        - gamma: Decay rate
        """
        self.states_names = states
        self.actions_names = actions
        self.states = list(range(len(states)))
        self.actions = list(range(len(actions)))
        self.alpha: float = 0.1
        self.gamma: float = 0.99
        self.epsilon = 1

        self.rl_parameters: List[str] = ['alpha', 'gamma']
        self.Q = np.zeros((len(self.states), len(self.actions)))
        self.current_state: State = random.choice(self.states)

    def best_action(self) -> Action:
        """ Chooses the best action to take in its current state from its Q matrix """
        return np.argmax(self.Q[self.current_state])

    def epsilon_greedy(self) -> Action:
        """ Choose an action with an epsilon-greedy policy """
        p = random.random()
        if p <= self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.best_action()
        self.epsilon *= self.gamma
        return action

    def policy(self) -> Action:
        return self.epsilon_greedy()

    def set_params(self, params: Dict[str, Any]) -> None:
        """ Set parameters of routine """
        if params is None:
            return
        for varname, value in params.items():
            if varname not in self.rl_parameters:
                raise Exception(f'{varname} is not a valid keyword argument')
            setattr(self, varname, value)

    @abstractmethod
    def perform_action(self, action: Action) -> State:
        """ Determines the target state after performing an action in the current state  """

    @abstractmethod
    def reward(self, reward_params: Dict[str, Any]):
        """ Defines the agent's reward """

    def update_Q(self, state: State, action: Action, next_state: State, reward: float):
        """ Updates Q-matrix (Q-Learning) """
        self.Q[state, action] += self.alpha * (reward + self.gamma*np.argmax(self.Q[next_state]) - self.Q[state, action])

    # @abstractmethod
    # def observe_state(self, state: State) -> None:
    #     """ Observe the state in which it is located """

    def update_state(self, state: State) -> None:
        """ Updates the current state after performing an action in its current state """
        self.current_state = state

    def display(self) -> DataFrame:
        df = DataFrame(data=self.Q, index=self.states_names, columns=self.actions_names)
        return df

    def display_info(self):
        print('Q Learning Params')
        print('alpha =', self.alpha)
        print('gamma =', self.gamma)
        print('epsilon =', self.epsilon)

    def __repr__(self):
        return str(self.Q)

    # TODO: improve this in new version
    def q_learn_step(self, reward_params):
        """ Generic Q-learning step, it should not be used in this version """
        action = self.policy()
        new_state = self.perform_action(action)
        # self.observe_state(new_state)
        reward = self.reward(reward_params)
        self.update_Q(state=self.current_state, action=action, next_state=new_state, reward=reward)
        self.update_state(new_state)


class NeighborhoodQLearning(QLearning):
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.N = self.agent.N
        self.neighborhoods = self.N.use_methods
        super().__init__(states=self.neighborhoods, actions=self.neighborhoods)
        self.reference_solution = self.agent.in_solution
        self.ref_cost = self.reference_solution.cost()

    def reward(self, solution: Solution) -> float:
        """ The reward is defined by how much the solution found improves the reference solution. """
        return self.ref_cost - solution.cost()

    def perform_action(self, action: Action) -> State:
        """ In this context, to choose an action means to use a particular neighborhood function """
        self.N.set_params({'use_methods': [self.neighborhoods[action]]})
        return action

    def explore(self, solution: Solution) -> Solution:
        """
        Each time an agent "explores", it uses one solution to find another, through a neighborhood structure, which it
        chose according to its "policy". In the process, it calculates its reward and updates its state and Q matrix.
        """
        action = self.policy()
        new_state = self.perform_action(action)
        new_sol = self.N(solution)
        reward = self.reward(new_sol)
        self.update_Q(state=self.current_state, action=action, next_state=new_state, reward=reward)
        self.update_state(new_state)
        return new_sol

    def display_info(self):
        super().display_info()
        print('Q-learning of choice of neighborhood structure sequences')
        print('states:', self.states_names)
        print('actions:', self.actions_names)
