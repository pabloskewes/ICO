from typing import Optional, Dict
import random

from solution import VRPTWSolution as Sol
from context import VRPTWContext
from metaheuristics.base_problem import Neighborhood, Solution


class VRPTWNeighborhood(Neighborhood):
    context: VRPTWContext = None

    def __init__(self, params: Optional[Dict] = None):
        """
        Initializes a solution neighborhood manager of a VRPTW.
        """
        super().__init__()

        self.verbose = 0
        self.init_sol = 'random'

        self.valid_params = ['init_sol', 'verbose']
        if params is not None:
            self.set_params(params)

    def __str__(self):
        return f"Neighborhood of params: verbose={self.verbose}, init_sol={self.init_sol}"

    def initial_solution(self) -> Solution:
        if self.init_sol == 'random':
            init_sol = VRPTWNeighborhood.random_solution(nb_cust=len(self.context.customers) - 1,
                                                         force_check_vrptw=self.context)
        elif isinstance(self.init_sol, Sol):
            init_sol = self.init_sol
        else:
            raise Exception('Not a valid form of initial solution')
        return init_sol

    def get_neighbor(self, solution) -> Solution:
        new_sol = self.shuffle(solution)
        return new_sol
    
    @staticmethod
    def random_solution(nb_cust, force_check_vrptw=None, verbose=0) -> Solution:
        """
        Generates a random pattern of numbers between 0 and nb_cust, in the form of a solution.
        :param nb_cust: Number of customers wanted in the solution to be generated
        :param force_check_vrptw: The default is None and does nothing. When delivering a VRPTW instance in this parameter,
        the legitimacy of the generated solution will be checked (using 'check_solution') based on the context of
        that particular VRPTW instance.
        :param verbose: Level of verbosity desired
        :return: Solution (or nothing)
        """
        numbers = list(range(1, nb_cust + 1))
        random.shuffle(numbers)
        proportion = random.choice([0.2, 0.3, 0.4])
        n_0 = int(nb_cust * proportion)
        zero_positions = []
        zero_pos_candidates = list(range(1, nb_cust - 1))
        for _ in range(n_0):
            if verbose >= 2:
                print('candidates:', zero_pos_candidates)
            try:
                zero_pos = random.choice(zero_pos_candidates)
            except IndexError:
                if verbose >= 1:
                    print('A problem ocurred, generating new random solution')
                return VRPTWNeighborhood.random_solution(nb_cust=nb_cust, force_check_vrptw=force_check_vrptw)
            if verbose >= 2:
                print('zero_pos chosen:', zero_pos)
            zero_pos_candidates = list(set(zero_pos_candidates) - {zero_pos, zero_pos + 1, zero_pos - 1})
            zero_positions.append(zero_pos)
        for pos in zero_positions:
            numbers.insert(pos, 0)
        solution = [0] + numbers + [0]
        string = str(solution).replace('0, 0, 0', '0').replace('0, 0', '0')
        code_solution = list(map(int, string.strip('][').split(',')))
        if code_solution[-1] != 0:
            code_solution.append(0)
        solution = Sol(code_solution)
        if force_check_vrptw:
            check = solution.checker()
            if not check:
                if verbose >= 1:
                    print('Solution generated is not legitimate, a new one will be created.')
                return VRPTWNeighborhood.random_solution(nb_cust=nb_cust, force_check_vrptw=force_check_vrptw)
            else:
                if verbose >= 1:
                    print(f'A legitimate solution was successfully generated:\n{solution}')
        return solution

    def shuffle(self, solution=None):
        """
        Apply the random_solution function to obtain a valid random solution (you can't even say it's really in the
        neighborhood of the solution, but it can be very useful).
        :param solution: Nothing, only for aesthetic purposes.
        :return:
        """
        r_sol = self.random_solution(nb_cust=len(self.context.customers)-1,
                                     force_check_vrptw=self.context, verbose=self.verbose)
        return r_sol





