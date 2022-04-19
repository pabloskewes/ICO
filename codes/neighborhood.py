from typing import Optional, Dict, Tuple, List
import random
from copy import deepcopy
from itertools import combinations
from math import factorial

from solution import VRPTWSolution as Sol
from solution import VRPTWSolution
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
        self.max_iter = 10
        self.force_new_sol = False
        self.use_methods = ['switch_two_customers_intra_route', 'inter_route_swap'
                            'switch_three_consecutive', 'reverse_a_sequence', 'crossover']

        self.valid_params = ['init_sol', 'verbose', 'choose_mode', 'use_methods', 'max_iter', 'force_new_sol']
        if params is not None:
            self.set_params(params)

    def __str__(self):
        return f"Neighborhood of params: verbose={self.verbose}, init_sol={self.init_sol}"

    def initial_solution(self) -> Solution:
        if self.init_sol == 'random':
            init_sol = self.random_solution(nb_cust=len(self.context.customers) - 1)
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

    def get_neighbor_from_two(self, solution1, solution2) -> Tuple[Solution, Solution]:

        if hasattr(self, self.choose_mode):
            new_sol1, new_sol2 = getattr(self, self.choose_mode)(solution1, solution2)
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
        r_sol = self.random_solution(nb_cust=len(self.context.customers) - 1)
        return r_sol

    # NEIGHBORHOOD FUNCTION
    def switch_two_customers_intra_route(self, solution: VRPTWSolution) -> VRPTWSolution:
        """
        Switches two random customers in one random route (except the first and last customers who are the depot),
        then returns new solution
        :param : solution
        """
        is_sol = False
        routes = solution.routes.copy()
        if self.verbose > 1:
            print(f"Voici les routes {routes}")
        breaker = 0  # in case there are no possible neighbors that are solutions with this function
        route = random.choice(routes)
        index_route = solution.routes.index(route)
        # Verification que la route choisie est assez longue (au moins 4 élements)
        if self.verbose > 1:
            print(f"Route premièrement choisie est {route} dont l'index est {index_route}")
        while len(route) < 4 and len(routes) > 0:
            if self.verbose > 1:
                print("changement de route")
            routes.remove(route)
            route = random.choice(routes)
            index_route = solution.routes.index(route)
            if len(routes) == 0:
                if self.verbose > 0:
                    print("No possibility to apply this neighborhood function")
                return solution
        if self.verbose > 1:
            print(f"Route sur laquelle on travaille est {route} et son index est {index_route}\n")

        # Etape de swap + vérification que la solution trouvée est bien une solution du problème
        while (not is_sol) and (breaker < 10):
            breaker += 1
            route_test = route.copy()
            i = random.randint(1, len(route_test) - 2)  # on prend un élément au hasard, extrémités exclues
            j = random.randint(1, len(route_test) - 2)  # idem
            # On vérifie que i et j sont différents
            while j == i:
                j = random.randint(1, len(route_test) - 2)
            route_test[i], route_test[j] = route_test[j], route_test[i]
            routes_copy = solution.routes
            routes_copy[index_route] = route_test
            sol_found = Sol(routes_copy)
            is_sol = sol_found.route_checker(route_test)
            if (not is_sol) and (self.verbose > 1):
                print("Solution trouvée non conforme\n")

        if not is_sol:
            if self.verbose > 0:
                print("No neighbor found that is a solution")
            return solution
        neighbor = Sol(routes_copy)
        return neighbor

    # NEIGHBORHOOD FUNCTION 2 - INTER ROUTE SWAP
    def inter_route_swap(self, solution: VRPTWSolution) -> VRPTWSolution:
        routes = solution.routes
        new_routes = deepcopy(routes)
        S = VRPTWSolution()
        is_sol = False
        used_route_pairs = []
        routes_iter = 0
        max_routes_iter = int(factorial(len(routes)) / (factorial(len(routes) - 2) * 2))
        while not is_sol and routes_iter < min(self.max_iter, max_routes_iter):
            random_routes = tuple(random.sample(list(enumerate(routes)), 2))
            index_r1, index_r2 = random_routes[0][0], random_routes[1][0]
            # noinspection PyTypeChecker
            route1: List[List[int]] = random_routes[0][1]
            # noinspection PyTypeChecker
            route2: List[List[int]] = random_routes[1][1]
            if {index_r1, index_r2} in used_route_pairs:
                routes_iter += 1
                continue

            used_cust_pairs = []
            cust_iter = 0
            max_cust_iter = (len(route1) - 2) * (len(route2) - 2)
            while not is_sol and cust_iter < min(self.max_iter, max_cust_iter):
                index_c1, cust1 = tuple(random.choice(list(enumerate(route1[1:-1]))))
                index_c2, cust2 = tuple(random.choice(list(enumerate(route2[1:-1]))))
                index_c1, index_c2 = index_c1 + 1, index_c2 + 1     # because of first 0 at route
                if {index_c1, index_c2} in used_cust_pairs:
                    cust_iter += 1
                    continue
                new_routes[index_r1][index_c1] = cust2
                new_routes[index_r2][index_c2] = cust1
                is_sol = S.route_checker(route1) and S.route_checker(route2)
                if is_sol:
                    new_sol = VRPTWSolution(new_routes)
                    if not self.force_new_sol or new_sol != solution:
                        return new_sol
                    is_sol = False
                used_cust_pairs.append({index_c1, index_c2})
                cust_iter += 1
                new_routes = deepcopy(routes)

            used_route_pairs.append({index_r1, index_r2})
            routes_iter += 1
        if self.verbose >= 1:
            print("inter_route_swap wasn't able to find a neighbor for this solution")
        return solution

    # NEIGHBORHOOD FUNCTION
    def switch_three_customers_intra_route(self, solution) -> Solution:
        """
        Switches three random customers in one random route (except the first and last customers who are the depot),
        then returns new solution
        :param : solution
        """
        is_sol = False
        routes = solution.routes.copy()
        if self.verbose > 1:
            print(f"Voici les routes {routes}")
        breaker = 0  # in case there are no possible neighbors that are solutions with this function
        route = random.choice(routes)
        index_route = solution.routes.index(route)
        # Verification que la route choisie est assez longue (au moins 4 élements)
        if self.verbose > 1:
            print(f"Route premièrement choisie est {route} dont l'index est {index_route}")
        while len(route) < 5 and len(routes) > 0:
            if self.verbose > 1:
                print("changement de route")
            routes.remove(route)
            route = random.choice(routes)
            index_route = solution.routes.index(route)
            if len(routes) == 0:
                if self.verbose > 0:
                    print("No possibility to apply this neighborhood function")
                return solution
        if self.verbose > 1:
            print(f"Route sur laquelle on travaille est {route} et son index est {index_route}\n")

        # Etape de swap + vérification que la solution trouvée est bien une solution du problème
        while (not is_sol) and (breaker < 10):
            breaker += 1
            route_test = route.copy()
            i = random.randint(1, len(route_test) - 2)  # on prend un élément au hasard, extrémités exclues
            j = random.randint(1, len(route_test) - 2)  # idem
            k = random.randint(1, len(route_test) - 2)  # idem
            # On vérifie que i et j sont différents
            while j == i or j == k or i == k:
                j = random.randint(1, len(route_test) - 2)
                k = random.randint(1, len(route_test) - 2)
            route_test[i], route_test[j], route_test[k] = route_test[k], route_test[i], route_test[j]
            routes_copy = solution.routes
            routes_copy[index_route] = route_test
            sol_found = Sol(routes_copy)
            is_sol = sol_found.route_checker(route_test)
            if (not is_sol) and (self.verbose > 1):
                print("Solution trouvée non conforme\n")

        if not is_sol:
            if self.verbose > 0:
                print("No neighbor found that is a solution")
            return solution
        neighbor = Sol(routes_copy)
        return neighbor

    # NEIGHBORHOOD FUNCTION
    def reverse_a_sequence(self, solution: VRPTWSolution):
        """
        Reverse a sequence of code by randomly choosing the start position and the end position in the solution_code (except the first,
        second to last and last customers), then returns new solution
        :param : solution
        """
        sol_code = solution.sol_code
        is_sol = False
        breaker = 0
        while (not is_sol) or (breaker < 100):
            breaker += 1
            head = random.randrange(1, len(sol_code) - 1)
            end = random.randrange(head, len(sol_code) - 1)
            tmp = sol_code[head:end]
            tmp.reverse()
            sol_code = sol_code[:head] + tmp + sol_code[end:]
            solution_found = Sol(sol_code)
            is_sol = all((solution_found.route_checker(route) for route in solution_found.routes))
        if (not is_sol) and (self.verbose > 1):
            print("No neighbor found that is a solution")
            return solution
        neighbor = Sol(sol_code)
        return neighbor

    # GA NEIGHBORHOOD FUNCTION
    def crossover(self, solution1: VRPTWSolution, solution2: VRPTWSolution) -> Tuple[VRPTWSolution, VRPTWSolution]:
        """
        Generate two solutions by combining sections from two different solutions (except the first,
        second to last and last customers), then returns new solutions
        :param : solution
        """
        sol_code1 = solution1.sol_code
        sol_code2 = solution2.sol_code

        is_sol = False
        breaker = 0
        sol_code = sol_code1
        while (not is_sol) or (breaker < 100):

            pos = random.randrange(1, min(len(sol_code1), len(sol_code2)) - 1)
            child1 = sol_code1[:pos] + sol_code2[pos:]
            child2 = sol_code2[:pos] + sol_code1[pos:]

            copy_child1 = deepcopy(child1)
            copy_child2 = deepcopy(child2)

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

            solution_found1 = Sol(child1)
            solution_found2 = Sol(child2)
            is_sol = all((solution_found1.route_checker(route) for route in solution_found1.routes)) and \
                     all((solution_found2.route_checker(route) for route in solution_found2.routes))

        if (not is_sol) and (self.verbose > 1):
            print("No neighbor found that is a solution")
            return solution1, solution2

        neighbor1 = Sol(child1)
        neighbor2 = Sol(child2)
        return neighbor1, neighbor2

    def random_solution(self, nb_cust) -> VRPTWSolution:
        """
        Generates a random pattern of numbers between 0 and nb_cust, in the form of a solution.
        :param nb_cust: Number of customers wanted in the solution to be generated
        :return: Solution (or nothing)
        """

        def simplify(L, simpleL=[], i=0, on_zero=False):
            if i >= len(L):
                return simpleL
            if L[i] == 0:
                if on_zero:
                    return simplify(L, simpleL, i+1, True)
                else:
                    return simplify(L, simpleL+[L[i]], i+1, True)
            else:
                return simplify(L, simpleL+[L[i]], i+1, False)

        numbers = list(range(1, nb_cust + 1))
        is_sol = False
        solution = None
        while not is_sol:
            random.shuffle(numbers)
            proportion = random.choice([0.05, 0.1, 0.15])
            n_0 = int(nb_cust * proportion)
            zero_positions = []
            zero_pos_candidates = list(range(1, nb_cust - 1))

            for _ in range(n_0):
                if self.verbose >= 2:
                    print('candidates:', zero_pos_candidates)
                try:
                    zero_pos = random.choice(zero_pos_candidates)
                except IndexError:
                    if self.verbose >= 1:
                        print('A problem ocurred, generating new random solution')
                    continue
                if self.verbose >= 2:
                    print('zero_pos chosen:', zero_pos)
                zero_pos_candidates = list(set(zero_pos_candidates) - {zero_pos, zero_pos + 1, zero_pos - 1})
                zero_positions.append(zero_pos)

            for pos in zero_positions:
                numbers.insert(pos, 0)
            solution = [0] + numbers + [0]
            code_solution = simplify(solution)
            if code_solution[-1] != 0:
                code_solution.append(0)
            solution = Sol(code_solution)

            check = solution.checker()
            if not check:
                if self.verbose >= 1:
                    print('Solution generated is not legitimate, a new one will be created.')
                continue
            else:
                is_sol = True
                if self.verbose >= 1:
                    print(f'A legitimate solution was successfully generated:\n{solution}')

        return solution

    # NEIGHBORHOOD FUNCTION 7 - DELETE SMALLEST ROUTE
    def delete_smallest_route(self, solution: VRPTWSolution) -> VRPTWSolution:
        """
        Deletes the smallest route and inserts its customers in the other routes
        :param solution
        :return: Solution (or nothing)
        """
        routes = deepcopy(solution.routes)
        lengths_list = list(map(len, routes))
        smallest_index = lengths_list.index(min(lengths_list))
        deleted_route = routes.pop(smallest_index)
        new_solution = None
        is_sol = False
        n_iter = 0
        while not is_sol and n_iter < self.max_iter:
            n_iter += 1
            for i in range(1, len(deleted_route)-1):
                r_route = random.randint(0, len(routes)-1)
                r_pos = random.randint(1, len(routes[r_route])-2)
                routes[r_route].insert(r_pos, deleted_route[i])
            new_solution = VRPTWSolution(routes)
            is_sol = all((new_solution.route_checker(route) for route in new_solution.routes))
            if is_sol:
                return new_solution
        return solution
