from metaheuristics.base_problem import Context, Solution, Neighborhood, SolutionSpace, Problem
from dataclasses import dataclass
from pandas import DataFrame, read_csv
import numpy as np
from typing import List
import random
import os

@dataclass
class FlowShopContext(Context):
    machines: np.array
    num_machines: int
    num_tasks: int

    def __repr__(self):
        col_names = ['T'+str(i) for i in range(1, self.num_tasks+1)]
        machines_names = ['M'+str(i) for i in range(1, self.num_machines+1)]
        return str(DataFrame(self.machines, columns=col_names, index=machines_names))


class FlowShopSolution(Solution):
    context: FlowShopContext = None

    def __init__(self, flowshop: List[int] = None, params=None):
        super().__init__()

        self.flowshop = np.array(flowshop)

        self.valid_params = []
        if params is not None:
            self.set_params(params)

    def __repr__(self):
        return str(self.flowshop)

    def __getitem__(self, key):
        return self.flowshop[key]

    def __setitem__(self, key, value):
        self.flowshop[key] = value

    def __len(self)__:
        return len(self.flowshop)

    def copy(self):
        return FlowShopSolution(self.flowshop)

    def orderMatrix(self):
        machines = self.context.machines
        num_tasks = self.context.num_tasks
        num_machines = self.context.num_machines
        new_matrix = np.zeros((num_machines, num_tasks), dtype=np.int32)
        for i in range(num_tasks):
            new_matrix[:, i] = machines[:, self.flowshop[i]]
        return new_matrix

    def checker(self):
        length = len(self.flowshop) == self.context.num_tasks
        all_presents = all([i in self.flowshop for i in range(self.context.num_tasks)])
        return length and all_presents

    def cost(self) -> float:
        """  Returns the cost of a solution """
        machines = self.context.machines
        solution = self.flowshop
        mat = self.orderMatrix()
        T_debut = np.zeros([len(machines),len(solution)])
        T_debut[1][0] = mat[0][0]
        T_debut[2][0] = T_debut[1][0] + mat[1][0]
        for i in range(1,len(solution)):
            T_debut[0][i] = T_debut[0][i-1] + mat[0][i-1]
        for i in [1,2]:
            for j in range(1,len(solution)):
                T_debut[i,j] = max(T_debut[i-1][j] + mat[i-1][j], T_debut[i][j-1]+ mat[i][j-1])
        return int(T_debut[-1,-1] + mat[-1][-1])


class FlowShopNeighborhood(Neighborhood):
    context: FlowShopContext = None

    def __init__(self, params=None):
        super().__init__()

        self.init_sol = 'random'
        self.verbose = 0
        self.choose_mode = 'random'
        self.use_methods = ['swap2',
                            'flip2']
        self.force_new_sol = True
        self.methods_ids = {i+1: method for i, method in enumerate(self.use_methods)}

        self.valid_params = ['init_sol', 'verbose', 'use_methods', 'choose_mode',
                            'force_new_sol']
        if params is not None:
            self.set_params(params)

    def initial_solution(self) -> Solution:
        if self.init_sol == 'random':
            flowshop = list(range(self.context.num_tasks))
            random.shuffle(flowshop)
            init_sol = FlowShopSolution(flowshop)
        elif isinstance(self.init_sol, FlowShopSolution):
            init_sol = self.init_sol
        else:
            raise Exception('Not a valid init sol')
        return init_sol

    def get_neighbor(self, solution: Solution) -> Solution:
        """
        Defines the way in which the neighborhood function to be used is chosen through the attribute "choose_mode".
        "choose_mode" can take the following values:
        - "random": chooses a random neighborhood from among those found in the "use_methods" attribute
        - "best": looks for a solution for each neighborhood in "use_methods" and returns the best one.
        - Directly the name of the method. Ex: "intra_route_swap".
        - A number between 1 and 8 representing the id of a neighborhood (encoded in the "methods_ids" attribute).
        :param solution: Solution for which a neighborhood is being sought
        :return: Neighbor solution found
        """
        if self.choose_mode == 'random':
            method_name = random.choice(self.use_methods)
            method_name = self.methods_ids[method_name] if type(method_name) == int else method_name
            new_sol = getattr(self, method_name)(solution)

        elif self.choose_mode == 'best':
            solutions_found = [getattr(self, method_name)(solution) for method_name in self.use_methods]
            best_solutions = list(map(lambda sol: sol.cost(), solutions_found))
            index = best_solutions.index(min(best_solutions))
            new_sol = solutions_found[index]

        elif hasattr(self, self.choose_mode):
            new_sol = getattr(self, self.choose_mode)(solution)

        elif type(self.choose_mode) == int:
            method_name = self.methods_ids[self.choose_mode]
            new_sol = getattr(self, method_name)(solution)

        else:
            raise Exception(f'"{self.choose_mode}" is not a valid parameter for choose_mode')
        return new_sol

    def swap2(self, solution):
        num_tasks = self.context.num_tasks
        new_solution = solution.copy()
        while solution.cost() == new_solution.cost():
            new_solution = solution.copy()
            r1 = random.randint(0, num_tasks-1)
            r2 = random.randint(0, num_tasks-1)
            while r1 == r2:
                r2 = random.randint(0, num_tasks-1)
            aux = new_solution[r1]
            new_solution[r1] = new_solution[r2]
            new_solution[r2] = aux
            if not self.force_new_sol:
                break

        return new_solution

    def flip2(self, solution):
        num_tasks = self.context.num_tasks
        new_solution = solution.copy()
        while solution.cost() == new_solution.cost():
            new_solution = solution.copy()
            r1 = random.randint(0, num_tasks-1)
            r2 = 0 if r1 == num_tasks-1 else r1 + 1
            aux = new_solution[r1]
            new_solution[r1] = new_solution[r2]
            new_solution[r2] = aux
            if not self.force_new_sol:
                break

        return new_solution



class FlowShopSolutionSpace(SolutionSpace):
    context: FlowShopContext = None

    def __init__(self, params=None):
        super().__init__()

        self.valid_params = []
        if params is not None:
            self.set_params(params)

    def distance(self, s1: Solution, s2: Solution) -> float:
        graph_1 = [(s1[i], s1[i+1]) for i in range(len(s1)-1)]
        graph_2 = [(s2[i], s2[i+1]) for i in range(len(s2)-1)]
        diff_set = set(graph_1)^set(graph_2)
        return len(diff_set)


class FlowShop(Problem):
    """  Flow Shop Scheduling Problem """
    solution = FlowShopSolution
    neighborhood = FlowShopNeighborhood
    solution_space = FlowShopSolutionSpace


def load_flowshop(filename) -> FlowShopContext:
    ROOT_DIR = os.path.abspath('..')
    DATA_DIR = os.path.join(ROOT_DIR, 'data_flowshop')
    DIR = os.path.join(DATA_DIR, filename)

    machines = read_csv(DIR, index_col=0).to_numpy()
    num_machines = machines.shape[0]
    num_tasks = machines.shape[1]

    return FlowShopContext(machines=machines, num_machines=num_machines, num_tasks=num_tasks)


def generate_flowshop_instance(filename, num_machines=6, num_tasks=9):
    tasks_name = ['T'+str(i) for i in range(num_tasks)]
    min_time = num_tasks*3
    max_time = num_tasks*10 + 1
    df = DataFrame(np.random.randint(low=min_time, high=max_time, size=(num_machines, num_tasks)),
                    columns=tasks_name)
    ROOT_DIR = os.path.abspath('..')
    DATA_DIR = os.path.join(ROOT_DIR, 'data_flowshop')
    DIR = os.path.join(DATA_DIR, filename)
    df.to_csv(DIR)
