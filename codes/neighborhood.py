from typing import Optional, Dict
import random
import copy

from solution import VRPTWSolution as Sol
from context import VRPTWContext
from solution import sol_to_list_routes
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
        self.choose_mode = 'random'
        self.use_methods = ['switch_two_consecutive', 'switch_two_random',
                            'switch_three_consecutive', 'switch_three_random',
                            'reverse_a_sequence','crossover']

        self.valid_params = ['init_sol', 'verbose', 'choose_mode', 'use_methods']
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
        if self.choose_mode == 'random':
            method_name = random.choice(self.use_methods)
            new_sol = getattr(self, method_name)(solution)
        elif self.choose_mode == 'best':
            solutions_found = [getattr(self, method_name)(solution) for method_name in self.use_methods]
            best_solutions = list(map(lambda sol: sol.cost(), solutions_found))
            index = best_solutions.index(min(best_solutions))
            new_sol = solutions_found[index]
        elif hasattr(self, self.choose_mode):
            new_sol = getattr(self, self.choose_mode)(solution)
        else:
            raise Exception(f'"{self.choose_mode}" is not a valid parameter for choose_mode')
        return new_sol

    def get_neighbor_from_two(self, solution1,solution2) -> Solution:

        if hasattr(self, self.choose_mode):
            new_sol1,new_sol2 = getattr(self, self.choose_mode)(solution1,solution2)
        else:
            raise Exception(f'"{self.choose_mode}" is not a valid parameter for choose_mode')
        return new_sol1, new_sol2

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

    def switch_two_consecutive(self, solution):
        """
        Switches two random consecutive customers in the solution_code (except the first,
        second to last and last customers), then returns new solution
        :param : solution
        """
        sol_code = solution.sol_code
        is_sol = False
        breaker = 0  # in case there are no possible neighbors that are solutions with this function
        while (not is_sol) and (breaker < 100):
            breaker += 1
            i = random.randint(1, len(sol_code)-3)
            sol_code[i], sol_code[i+1] = sol_code[i+1], sol_code[i]
            solution_found = Sol(sol_code)
            is_sol = all((solution_found.route_checker(route) for route in solution_found.routes))
        if (not is_sol) and (self.verbose > 1):
            print("No neighbor found that is a solution")
            return solution
        neighbor = Sol(sol_code)
        return neighbor

    def switch_two_random(self, solution):
        """
        Switches two random random customers in the solution_code (except the first,
        second to last and last customers), then returns new solution
        :param : solution
        """
        sol_code = solution.sol_code
        is_sol = False
        breaker = 0  # in case there are no possible neighbors that are solutions with this function
        while (not is_sol) and (breaker < 100):
            breaker += 1
            i = random.randint(1, len(sol_code)-3)
            j = random.randint(1, len(sol_code) - 3)
            sol_code[i], sol_code[j] = sol_code[j], sol_code[i]
            solution_found = Sol(sol_code)
            is_sol = all((solution_found.route_checker(route) for route in solution_found.routes))
        if (not is_sol) and (self.verbose > 1):
            print("No neighbor found that is a solution")
            return solution
        neighbor = Sol(sol_code)
        return neighbor

    def switch_three_consecutive(self, solution):
        """
        Switches three random consecutive customers in the solution_code (except the first,
        second to last and last customers), then returns new solution
        :param : solution
        """
        sol_code = solution.sol_code
        is_sol = False
        breaker = 0  # in case there are no possible neighbors that are solutions with this function
        while (not is_sol) and (breaker < 100):
            breaker += 1
            i = random.randint(1, len(sol_code)-4)
            sol_code[i], sol_code[i+1], sol_code[i+2] = sol_code[i+2], sol_code[i], sol_code[i+1]
            solution_found = Sol(sol_code)
            is_sol = all((solution_found.route_checker(route) for route in solution_found.routes))
        if (not is_sol) and (self.verbose > 1):
            print("No neighbor found that is a solution")
            return solution
        neighbor = Sol(sol_code)
        return neighbor

    def switch_three_random(self, solution):
        """
        Switches three random consecutive customers in the solution_code (except the first,
        second to last and last customers), then returns new solution
        :param : solution
        """
        sol_code = solution.sol_code
        is_sol = False
        breaker = 0  # in case there are no possible neighbors that are solutions with this function
        while (not is_sol) and (breaker < 100):
            breaker += 1
            i = random.randint(1, len(sol_code)-4)
            j = random.randint(1, len(sol_code) - 4)
            k = random.randint(1, len(sol_code) - 4)
            sol_code[i], sol_code[j], sol_code[k] = sol_code[k], sol_code[i], sol_code[j]
            solution_found = Sol(sol_code)
            is_sol = all((solution_found.route_checker(route) for route in solution_found.routes))
        if (not is_sol) and (self.verbose > 1):
            print("No neighbor found that is a solution")
            return solution
        neighbor = Sol(sol_code)
        return neighbor

    def reverse_a_sequence(self,solution):
        '''
        Reverse a sequence of code by randomly chooing the start position and the end position in the solution_code (except the first,
        second to last and last customers), then returns new solution
        :param : solution
        '''
        sol_code = solution.sol_code
        is_sol = False
        breaker = 0  
        while(not is_sol) or (breaker<100):
            breaker += 1
            head = random.randrange(1, len(sol_code)-1)
            end = random.randrange(head, len(sol_code)-1)
            tmp = sol_code[head:end]
            tmp.reverse()
            sol_code = sol_code[:head] + tmp + sol_code[end:]
            solution_found= Sol(sol_code)
            is_sol = all((solution_found.route_checker(route) for route in solution_found.routes))
        if (not is_sol) and (self.verbose > 1):
            print("No neighbor found that is a solution")
            return solution
        neighbor = Sol(sol_code)
        return neighbor

    def crossover(self,solution1,solution2):
        '''
        Generate two solutions by combining sections from two different solutions (except the first,
        second to last and last customers), then returns new solutions
        :param : solution
        '''
        sol_code1=solution1.sol_code
        sol_code2=solution2.sol_code
        
        is_sol = False
        breaker = 0  
        sol_code=sol_code1
        while(not is_sol) or (breaker<100):
            
            pos = random.randrange(1, min(len(sol_code1),len(sol_code2)) - 1)
            child1 = sol_code1[:pos] + sol_code2[pos:]
            child2 = sol_code2[:pos] + sol_code1[pos:]
            
            copy_child1=copy.deepcopy(child1)
            copy_child2=copy.deepcopy(child2)

            count1 = 0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:  # If need to fix repeated gen
                    count2 = 0
                    for gen2 in sol_code1[pos:]:  # Choose next available gen
                        if gen2 not in copy_child1:
                            child1[count1] = sol_code1[pos:][count2]
                        count2 += 1
                count1 += 1
            count1 = 0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:  # If need to fix repeated gen
                    count2 = 0
                    for gen2 in sol_code2[pos:]:  # Choose next available gen
                        if gen2 not in copy_child2:
                            child2[count1] = sol_code2[pos:][count2]
                        count2 += 1
                count1 += 1

            solution_found1= Sol(child1)
            solution_found2= Sol(child2)
            is_sol = all((solution_found1.route_checker(route) for route in solution_found1.routes)) and  all((solution_found2.route_checker(route) for route in solution_found2.routes))

        if (not is_sol) and (self.verbose > 1):
            print("No neighbor found that is a solution")
            return solution1,solution2

        neighbor1 = Sol(child1)
        neighbor2 = Sol(child2)
        return neighbor1,neighbor2

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
