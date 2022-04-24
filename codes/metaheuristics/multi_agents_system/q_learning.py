import numpy as np
from pandas import DataFrame
from typing import List, Any, NewType
import random
from abc import ABC, abstractmethod

from ..base_problem import Solution, Neighborhood

State = Any
Action = Any


class QLearning(ABC):
    def __init__(self, states: List[State], actions: List[Action], alpha: float = 0.1, gamma: float = 0.9):
        """
        Initializes the main components of a Q-learning:
        - states: States in which an agent can be found
        - actions: Actions that an agent can perform, and lead to another state.
        - alpha: Learning Rate
        - gamma: Decay rate
        """
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1

        self.Q = np.zeros((len(self.states), len(self.actions)))
        self.current_state: State = random.choice(states)

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
    # @abstractmethod
    # def perform_action(self, action: Action) -> State:
    #     """ Determines the target state after performing an action in the current state  """
    #
    # def update_state(self, state: State) -> None:
    #     """ Updates the current state after performing an action in its current state """
    #     self.current_state = state

    @abstractmethod
    def reward(self, *args):
        """ Defines the agent's reward """

    def __repr__(self) -> DataFrame:
        df = DataFrame(data=self.Q, index=self.states, columns=self.actions)
        return df

    def update_Q(self, *args):
        """ Updates Q-matrix (Q-Learning) """
        r = self.reward(*args)


class NeighborhoodQLearning(QLearning):
    def __init__(self, neighborhood: Neighborhood, reference_solution: Solution, alpha: float = 0.1, gamma: float = 0.9):
        self.N = neighborhood
        self.neighborhoods = self.N.use_methods
        super().__init__(states=self.neighborhoods, actions=self.neighborhoods, alpha=alpha, gamma=gamma)
        self.reference_solution = reference_solution
        self.ref_cost = self.reference_solution.cost()

    # def perform_action(self, action: Action) -> State:
    #     """
    #     The neighborhood function to be used in N is set according to the chosen action, then the action itself is
    #     returned, since in this context it is equivalent to the following state.
    #     """
    #     self.N.set_params({'use_methods': [action]})
    #     return action

    def reward(self, solution: Solution) -> float:
        return self.ref_cost - solution.cost()

    def choose_neighborhood(self):
        action = self.policy()
        self.N.set_params({'use_methods': [action]})


