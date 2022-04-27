from .base_metaheuristic import BaseMetaheuristic
from .base_problem import Problem, Solution
from .multi_agents_system.q_learning import NeighborhoodQLearning
from .multi_agents_system.agents_factory import AgentFactory
from .multi_agents_system.multi_agent_system import MultiAgentSystem as MAS
from .multi_agents_system.sequential_models import SequentialModel
from tqdm import tqdm, trange
from copy import deepcopy

class AdaptativeLocalSearchQLearning(BaseMetaheuristic):

    def __init__(self, model=None, q_learning_params=None, max_iter_model=10,
                  max_iter_search=100, progress_bar=False, verbose=0,
                  solution_params=None, neighborhood_params=None,
                  solution_space_params=None):

        super().__init__()
        self.max_iter_search = max_iter_search
        self.progress_bar = progress_bar
        self.params = {'solution': solution_params,
                       'neighborhood': neighborhood_params,
                       'solution_space': solution_space_params}

        self.agent_structure = AgentFactory()(reinforced_learning=NeighborhoodQLearning,
                                              features_params={'reinforced_learning': q_learning_params})
        self.mas = MAS(model=model, agents = {self.agent_structure: 1},
                         max_iter=max_iter_model, pools=[], verbose=verbose,
                         solution_params=solution_params, neighborhood_params=neighborhood_params,
                         solution_space_params=solution_space_params)
        self.agent = None
        self.qlearning = None
        self.initial_solution = None
        self.next_state = None

    def fit(self, problem: Problem):
        instance = super().fit(problem)
        self.mas = self.mas.fit(problem)
        self.mas.setup()
        self.agent = self.mas.get_agents()[0]
        self.qlearning = self.agent.rl
        return instance

    def search(self) -> Solution:
        max_iter = 1000
        self.best_solution = self.qlearning.reference_solution
        pbar = tqdm(total=max_iter) if self.progress_bar else None
        if self.progress_bar:
            pbar.set_description('Cost: %.2f' %self.best_solution.cost())

        for _ in range(max_iter):
            self.local_search()
            if self.progress_bar:
                pbar.update()
                pbar.set_description('Cost: %.2f' %self.best_solution.cost())

        if self.progress_bar:
            pbar.close()
        return self.best_solution

    def local_search(self):
        improved = True
        episode = True
        no_improvement = 0
        self.actual_solution = self.best_solution
        self.current_state = 0
        self.last_current_state = self.current_state
        n_iter = 0
        total_iter = 0

        while improved:
            n_iter += 1
            episode_iter = 0
            reward = 0
            states_visited = deepcopy(self.qlearning.states)
            states_visited_count = 0
            self.current_state = self.qlearning.random_action()

            action = self.qlearning.best_action()
            self.current_state = self.qlearning.perform_action(action)
            self.actual_solution = self.qlearning.N(self.actual_solution)

            if self.actual_solution.cost() < self.best_solution.cost():
                self.best_solution = self.actual_solution
                reward = self.qlearning.reward(self.actual_solution)
            else:
                states_visited_count += 1
                states_visited[self.current_state] = -1

            while episode:
                total_iter += 1
                episode_iter += 1
                if no_improvement == 0:
                    self.last_current_state = self.current_state
                    self.current_state = self.qlearning.epsilon_greedy()
                else:
                    self.current_state = self.qlearning.random_action()

                action = self.qlearning.best_action()
                self.current_state = self.qlearning.perform_action(action)
                self.actual_solution = self.qlearning.N(self.actual_solution)

                if self.actual_solution.cost() < self.best_solution.cost():
                    # print('learning...')
                    self.best_solution = self.actual_solution
                    improved = True
                    episode = False
                    no_improvement = 0
                    reward += self.qlearning.reward(self.actual_solution)
                    # if self.progress_bar:
                    #     pbar.reset()
                    #     pbar.set_description('Cost: %.2f' %self.best_solution.cost())
                else:
                    no_improvement += 1
                    if states_visited[self.current_state] != -1:
                        states_visited_count += 1
                        states_visited[self.current_state] = -1
                    if no_improvement > self.max_iter_search or states_visited_count == len(self.qlearning.states):
                       improved = False
                       episode = False
                self.qlearning.update_Q(self.last_current_state,
                                        self.current_state,
                                        self.current_state,
                                        reward)
            episode = True
            self.qlearning.decrease_epsilon()
                # print(states_visited)
        # print('episodes:', n_iter)
        # print('iter episode', episode_iter)
        # print('total iterations:', total_iter)
