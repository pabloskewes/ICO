import sys
from typing import Optional, Dict, Tuple, List
import random
from copy import deepcopy
from itertools import combinations, product, permutations

from solution import VRPTWSolution as Sol
from solution import VRPTWSolution
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
        self.choose_mode = 'random'
        self.max_iter = 10
        self.force_new_sol = False
        self.use_methods = ['intra_route_swap', 'inter_route_swap',
                            'intra_route_shift', 'inter_route_shift',
                            'two_intra_route_swap', 'two_intra_route_shift',
                            'delete_smallest_route', 'delete_random_route']
        self.methods_ids = {i+1: method for i, method in enumerate(self.use_methods)}
        self.full_search = False
        self.best_neighbor = None
        self.best_cost = None

        self.valid_params = ['init_sol', 'verbose', 'choose_mode', 'use_methods',
                             'max_iter', 'force_new_sol', 'full_search']
        if params is not None:
            self.set_params(params)

    def __str__(self):
        return f"Neighborhood of params: verbose={self.verbose}, init_sol={self.init_sol}"

    def initial_solution(self) -> Solution:
        """
        Defines how an initial solution is chosen from the 'init_sol' parameter.
        It can take the values 'random' or directly a solution
        """
        if self.init_sol == 'random':
            init_sol = self.random_solution(nb_cust=len(self.context.customers) - 1)
        elif self.init_sol == 'trivial':
            init_sol = self.trivial_solution(len(self.context.customers) - 1)
        elif isinstance(self.init_sol, Sol):
            init_sol = self.init_sol
        else:
            raise Exception('Not a valid form of initial solution')
        return init_sol

    def get_neighbor(self, solution) -> Solution:
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

        elif type(self.choose_mode) == int:
            method_name = self.methods_ids[self.choose_mode]
            new_sol = getattr(self, method_name)(solution)

        elif hasattr(self, self.choose_mode):
            new_sol = getattr(self, self.choose_mode)(solution)

        else:
            raise Exception(f'"{self.choose_mode}" is not a valid parameter for choose_mode')
        return new_sol

    def get_neighbor_from_two(self, solution1, solution2) -> Tuple[Solution, Solution]:
        """
        Similar to "get_neighbor" but gets 2 solutions instead of one. Useful for metaheuristics that require
        neighborhoods using two solutions (generic algorithm for example).
        """
        if hasattr(self, self.choose_mode):
            new_sol1, new_sol2 = getattr(self, self.choose_mode)(solution1, solution2)
        else:
            raise Exception(f'"{self.choose_mode}" is not a valid parameter for choose_mode')
        return new_sol1, new_sol2

    # NEIGHBORHOOD FUNCTION 0 - SHUFFLE
    def shuffle(self, solution=None):
        """
        Apply the random_solution function to obtain a valid random solution (you can't even say it's really in the
        neighborhood of the solution, but it can be very useful).
        :param solution: Nothing, only for aesthetic purposes.
        :return: random solution
        """
        r_sol = self.random_solution(nb_cust=len(self.context.customers) - 1)
        return r_sol

    # NEIGHBORHOOD FUNCTION 1 - INTRA ROUTE SWAP
    def intra_route_swap(self, solution: VRPTWSolution) -> VRPTWSolution:
        """ Exchange 2 customers randomly on the same randomly chosen route """
        if self.full_search: self.set_tracker(solution)
        routes = solution.routes
        new_routes = deepcopy(routes)
        S = VRPTWSolution()
        is_sol = False

        available_routes_index = list(range(len(routes)))
        routes_iter = 0
        while not is_sol and routes_iter < self.max_iter and available_routes_index:
            r_index = random.choice(available_routes_index)
            available_routes_index.remove(r_index)

            route: List[int] = routes[r_index].copy()
            if len(route) < 4:
                continue
            c_index_pairs = list(combinations(range(1, len(route)-1), r=2))
            cust_iter = 0
            while not is_sol and cust_iter < self.max_iter and c_index_pairs:
                index_c1, index_c2 = random.choice(c_index_pairs)
                c_index_pairs.remove((index_c1, index_c2))

                # Swapping customers of same route
                route[index_c1], route[index_c2] = route[index_c2], route[index_c1]
                is_sol = S.route_checker(route)
                if is_sol:
                    new_routes[r_index] = route.copy()
                    new_sol = VRPTWSolution(new_routes)
                    if not self.full_search:
                        return new_sol
                    self.track_best(new_sol)
                    new_routes = deepcopy(routes)
                    is_sol = False

                cust_iter += 1
                route = routes[r_index].copy()
            routes_iter += 1

        if self.verbose >= 1 and not self.full_search:
            print("intra_route_swap wasn't able to find a neighbor for this solution")
        new_sol = self.best_neighbor if self.full_search else solution
        return new_sol

    # NEIGHBORHOOD FUNCTION 2 - INTER ROUTE SWAP
    def inter_route_swap(self, solution: VRPTWSolution) -> VRPTWSolution:
        """  Exchanges 2 clients between 2 random routes """
        if self.full_search: self.set_tracker(solution)
        routes = solution.routes
        new_routes = deepcopy(routes)
        S = VRPTWSolution()
        is_sol = False

        route_pairs = list(combinations(range(len(routes)), r=2))
        routes_iter = 0
        while not is_sol and routes_iter < self.max_iter and route_pairs:
            index_r1, index_r2 = random.choice(route_pairs)
            route_pairs.remove((index_r1, index_r2))
            route1, route2 = routes[index_r1].copy(), routes[index_r2].copy()

            cust_pairs = list(product(route1[1:-1], route2[1:-1]))
            cust_iter = 0
            while not is_sol and cust_iter < self.max_iter and cust_pairs:
                cust1, cust2 = random.choice(cust_pairs)
                cust_pairs.remove((cust1, cust2))
                index_c1, index_c2 = route1.index(cust1), route2.index(cust2)

                # Swapping customers of routes 1 and route 2
                new_routes[index_r1][index_c1] = cust2
                new_routes[index_r2][index_c2] = cust1
                is_sol = S.route_checker(new_routes[index_r1]) and S.route_checker(new_routes[index_r2])
                if is_sol:
                    new_sol = VRPTWSolution(new_routes)
                    if self.full_search:
                        self.track_best(new_sol)
                        new_routes = deepcopy(routes)
                        is_sol = False
                        continue
                    if not self.force_new_sol or new_sol != solution:
                        return new_sol
                    is_sol = False

                new_routes = deepcopy(routes)
                cust_iter += 1
            routes_iter += 1

        if self.verbose >= 1 and not self.full_search:
            print("inter_route_swap wasn't able to find a neighbor for this solution")
        new_sol = self.best_neighbor if self.full_search else solution
        return new_sol

    # NEIGHBORHOOD FUNCTION 3 - INTRA ROUTE SHIFT
    def intra_route_shift(self, solution: VRPTWSolution) -> VRPTWSolution:
        """ Moves a random client to another position on its same route """
        if self.full_search: self.set_tracker(solution)
        routes = solution.routes
        new_routes = deepcopy(routes)
        S = VRPTWSolution()
        is_sol = False

        available_routes_index = list(range(len(routes)))
        routes_iter = 0
        while not is_sol and routes_iter < self.max_iter and available_routes_index:
            r_index = random.choice(available_routes_index)
            available_routes_index.remove(r_index)

            route: List[int] = routes[r_index].copy()
            if len(route) < 4:
                continue
            c_positions = list(permutations(range(1, len(route) - 1), r=2))
            cust_iter = 0
            while not is_sol and cust_iter < self.max_iter and c_positions:
                index_c, new_position = random.choice(c_positions)
                c_positions.remove((index_c, new_position))

                # moving customer from one position to another in its same route
                cust = route.pop(index_c)
                route.insert(new_position, cust)
                is_sol = S.route_checker(route)
                if is_sol:
                    new_routes[r_index] = route.copy()
                    new_sol = VRPTWSolution(new_routes)
                    if not self.full_search:
                        return new_sol
                    self.track_best(new_sol)
                    new_routes = deepcopy(routes)
                    is_sol = False

                cust_iter += 1
                route = routes[r_index].copy()
            routes_iter += 1
        if self.verbose >= 1 and not self.full_search:
            print("intra_route_shift wasn't able to find a neighbor for this solution")
        new_sol = self.best_neighbor if self.full_search else solution
        return new_sol

    # NEIGHBORHOOD FUNCTION 4 - INTER ROUTE SHIFT
    def inter_route_shift(self, solution: VRPTWSolution) -> VRPTWSolution:
        """ Moves a random client to another position on another route """
        if self.full_search: self.set_tracker(solution)
        routes = solution.routes
        new_routes = deepcopy(routes)
        S = VRPTWSolution()
        is_sol = False

        available_routes_index = list(range(len(routes)))
        routes_iter = 0
        while not is_sol and routes_iter < self.max_iter and available_routes_index:
            r_index = random.choice(available_routes_index)
            available_routes_index.remove(r_index)

            route: List[int] = routes[r_index].copy()
            customers_index = range(1, len(route) - 1)
            other_routes_index = list(set(range(len(routes))) - {r_index})
            possible_positions = []
            for i in other_routes_index:
                other_route = routes[i].copy()
                possible_positions.extend(list(product([i], range(1, len(other_route) - 1))))
            c_combinations = list(product(customers_index, possible_positions))
            cust_iter = 0
            while not is_sol and cust_iter < self.max_iter and c_combinations:
                index_c, new_position = random.choice(c_combinations)
                new_position_route, new_position_cust = new_position
                c_combinations.remove((index_c, new_position))

                # inserting customer in another route
                cust = new_routes[r_index].pop(index_c)
                new_routes[new_position_route].insert(new_position_cust, cust)
                is_sol = S.route_checker(new_routes[r_index]) and S.route_checker(new_routes[new_position_route])
                if is_sol:
                    while [0, 0] in new_routes:
                        new_routes.remove([0, 0])
                    new_sol = VRPTWSolution(new_routes)
                    if not self.full_search:
                        return new_sol
                    self.track_best(new_sol)
                    is_sol = False
                cust_iter += 1
                new_routes = deepcopy(routes)
            routes_iter += 1
        if self.verbose >= 1 and not self.full_search:
            print("inter_route_shift wasn't able to find a neighbor for this solution")
        new_sol = self.best_neighbor if self.full_search else solution
        return new_sol

    # NEIGHBORHOOD FUNCTION 5 - INTRA ROUTE SWAP
    def two_intra_route_swap(self, solution: VRPTWSolution) -> VRPTWSolution:
        """ Exchange 2 consecutive customers randomly on the same randomly chosen route for another pair"""
        if self.full_search: self.set_tracker(solution)
        routes = solution.routes
        new_routes = deepcopy(routes)
        S = VRPTWSolution()
        is_sol = False

        available_routes_index = list(range(len(routes)))
        routes_iter = 0
        while not is_sol and routes_iter < self.max_iter and available_routes_index:
            r_index = random.choice(available_routes_index)
            available_routes_index.remove(r_index)

            route: List[int] = routes[r_index].copy()
            if len(route) < 6:
                continue
            c_positions = list(permutations(range(1, len(route) - 2), r=2))
            c_positions = list(filter(lambda pos: pos[0]+1 != pos[1] and pos[0]-1 != pos[1], c_positions))
            cust_iter = 0
            while not is_sol and cust_iter < self.max_iter and c_positions:
                index_pair_1, index_pair_2 = random.choice(c_positions)
                c_positions.remove((index_pair_1, index_pair_2))

                # Swapping customers of same route
                route[index_pair_1], route[index_pair_2] = route[index_pair_2], route[index_pair_1]
                route[index_pair_1 + 1], route[index_pair_2 + 1] = route[index_pair_2 + 1], route[index_pair_1 + 1]
                is_sol = S.route_checker(route)
                if is_sol:
                    new_routes[r_index] = route.copy()
                    new_sol = VRPTWSolution(new_routes)
                    if not self.full_search:
                        return new_sol
                    self.track_best(new_sol)
                    new_routes = deepcopy(routes)
                    is_sol = False
                cust_iter += 1
                route = routes[r_index].copy()
            routes_iter += 1
        if self.verbose >= 1 and not self.full_search:
            print("intra_route_swap wasn't able to find a neighbor for this solution")
        new_sol = self.best_neighbor if self.full_search else solution
        return new_sol

    # NEIGHBORHOOD FUNCTION 6 - TWO INTRA ROUTE SHIFT
    def two_intra_route_shift(self, solution: VRPTWSolution) -> VRPTWSolution:
        """ Moves a random client to another position on its same route """
        if self.full_search: self.set_tracker(solution)
        routes = solution.routes
        new_routes = deepcopy(routes)
        S = VRPTWSolution()
        is_sol = False

        available_routes_index = list(range(len(routes)))
        routes_iter = 0
        while not is_sol and routes_iter < self.max_iter and available_routes_index:
            r_index = random.choice(available_routes_index)
            available_routes_index.remove(r_index)

            route: List[int] = routes[r_index].copy()
            if len(route) < 5:
                continue

            c_positions = list(permutations(range(1, len(route) - 2), r=2))
            c_positions = list(filter(lambda pos: pos[0]+1 != pos[1], c_positions))
            cust_iter = 0
            while not is_sol and cust_iter < self.max_iter and c_positions:
                index_c, new_position = random.choice(c_positions)
                c_positions.remove((index_c, new_position))

                # moving 2 consecutive customers from one position to another in its same route
                cust_1 = route.pop(index_c)
                cust_2 = route.pop(index_c)
                route.insert(new_position, cust_2)
                route.insert(new_position, cust_1)
                is_sol = S.route_checker(route)
                if is_sol:
                    new_routes[r_index] = route.copy()
                    new_sol = VRPTWSolution(new_routes)
                    if not self.full_search:
                        return new_sol
                    self.track_best(new_sol)
                    new_routes = deepcopy(routes)
                    is_sol = False
                cust_iter += 1
                route = routes[r_index].copy()
            routes_iter += 1
        if self.verbose >= 1 and not self.full_search:
            print("two_intra_route_shift wasn't able to find a neighbor for this solution")
        new_sol = self.best_neighbor if self.full_search else solution
        return new_sol

    # NEIGHBORHOOD FUNCTION 7 - DELETE SMALLEST ROUTE
    def delete_smallest_route(self, solution: VRPTWSolution) -> VRPTWSolution:
        """
        Deletes the smallest route and inserts its customers in the other routes
        :param solution
        :return: Solution (or nothing)
        """
        if self.full_search: self.full_search = False
        if self.full_search: self.set_tracker(solution)
        routes = deepcopy(solution.routes)
        lengths_list = list(map(len, routes))
        smallest_route = min(lengths_list)
        S = VRPTWSolution()
        new_solution = None
        is_sol = False
        n_iter = 0
        available_routes = [i for i, route in enumerate(routes) if len(route) == smallest_route]

        while not is_sol and n_iter < self.max_iter and available_routes:
            n_iter += 1
            deleted_index = random.choice(available_routes)
            deleted_route = routes.pop(deleted_index)

            n_cycle = 0
            new_routes = deepcopy(routes)
            available_positions = [(i, j) for i in range(len(new_routes)) for j in range(1, len(new_routes[i]))]
            while n_cycle < self.max_iter and available_positions:
                n_cycle += 1
                for i in range(1, len(deleted_route)-1):
                    if not available_positions: break
                    r = random.choice(available_positions)
                    r_route = r[0]
                    r_pos = r[1]
                    new_routes[r_route].insert(r_pos, deleted_route[i])
                    available_positions.remove(r)
                is_sol = all((S.route_checker(route) for route in new_routes))
                if is_sol:
                    new_sol = VRPTWSolution(new_routes)
                    if not self.full_search:
                        return new_sol
                    self.track_best(new_sol)
                    is_sol = False

            routes.insert(deleted_index, deleted_route)
            available_routes.remove(deleted_index)
        if self.verbose >= 1 and not self.full_search:
            print("delete_smallest_route wasn't able to find a neighbor for this solution")
        new_sol = self.best_neighbor if self.full_search else solution
        return new_sol

    # NEIGHBORHOOD FUNCTION 8 - DELETE RANDOM ROUTE
    def delete_random_route(self, solution: VRPTWSolution) -> VRPTWSolution:
        """
        Deletes the random route and inserts its customers in the other routes
        :param solution
        :return: Solution (or nothing)
        """
        if self.full_search: self.full_search = False
        if self.full_search: self.set_tracker(solution)
        routes = deepcopy(solution.routes)
        new_solution = None
        S = VRPTWSolution()
        is_sol = False
        n_iter = 0
        available_routes = list(range(len(routes)))

        while not is_sol and n_iter < self.max_iter and available_routes:
            n_iter += 1
            deleted_index = random.choice(available_routes)
            deleted_route = routes.pop(deleted_index)

            n_cycle = 0
            new_routes = deepcopy(routes)
            available_positions = [(i, j) for i in range(len(new_routes)) for j in range(1, len(new_routes[i]))]
            while n_cycle < self.max_iter and available_positions:
                n_cycle += 1
                for i in range(1, len(deleted_route)-1):
                    if not available_positions: break
                    r = random.choice(available_positions)
                    r_route = r[0]
                    r_pos = r[1]
                    new_routes[r_route].insert(r_pos, deleted_route[i])
                    available_positions.remove(r)
                    # available_positions = [(i, j) for i in range(len(new_routes)) for j in range(1, len(new_routes[i]))]# this doesn't go well with simulated annealing
                is_sol = all((S.route_checker(route) for route in new_routes))
                if is_sol:
                    new_sol = VRPTWSolution(new_routes)
                    if not self.full_search:
                        return new_sol
                    self.track_best(new_sol)
                    is_sol = False

            routes.insert(deleted_index, deleted_route)
            available_routes.remove(deleted_index)
        if self.verbose >= 1 and not self.full_search:
            print("delete_random_route wasn't able to find a neighbor for this solution")
        new_sol = self.best_neighbor if self.full_search else solution
        return new_sol

    # GA NEIGHBORHOOD FUNCTION 1 - REVERSE A SEQUENCE
    def reverse_a_sequence(self, solution: VRPTWSolution):
        """
        Reverse a sequence of code by randomly choosing the start position and the end position in the solution_code
        (except the first, second to last and last customers), then returns new solution
        :param : solution
        """
        sol_code = solution.sol_code
        is_sol = False
        breaker = 0
        while (not is_sol) or (breaker < 10):
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

    # GA NEIGHBORHOOD FUNCTION 2 - CROSSOVER
    def crossover(self, solution1: VRPTWSolution, solution2: VRPTWSolution) -> Tuple[VRPTWSolution, VRPTWSolution]:
        """
        Generate two solutions by combining sections from two different solutions, then returns new solutions
        :param : solution
        """

        if not solution1.all_customers_checker() and solution2.all_customers_checker():
            return solution1, solution2

        n_iter=0
        is_sol=False
        sol_code1 = solution1.sol_code
        sol_code2 = solution2.sol_code

        list_cust_code=[]
        for i in sol_code1:
            if i!=0:
                list_cust_code.append(i)

        while not is_sol and n_iter < self.max_iter:
            n_iter += 1
            pos = random.randrange(1, (len(sol_code1)-sol_code1.count(0)) - 1)

            segment_left_1=[]
            segment_right_1=[]
            segment_left_2=[]
            segment_right_2=[]

            pos1=deepcopy(pos)
            for i in range(len(sol_code1)):
                if sol_code1[i] != 0:
                    pos1-=1
                segment_left_1.append(sol_code1[i])
                if pos1 ==0:
                    break
            segment_right_1=sol_code1[i+1:]

            pos2=deepcopy(pos)
            for i in range(len(sol_code2)):
                if sol_code2[i] != 0:
                    pos2-=1
                segment_left_2.append(sol_code2[i])
                if pos2 ==0:
                    break
            segment_right_2=sol_code2[i+1:]

            child1=segment_left_1+segment_right_2
            child2=segment_left_2+segment_right_1

            cust_child1=[]
            cust_repeat_child1=[]
            cust_miss_child1=[]

            for i in range(len(child1)):
                if child1[i]!=0:
                    if child1[i] not in cust_child1:
                        cust_child1.append(child1[i])
                    else:
                        cust_repeat_child1.append(child1[i])

            for  i in list_cust_code:
                if i not in cust_child1:
                    cust_miss_child1.append(i)

            cust_child1=[]
            j=0
            for i in range(len(child1)):
                if  child1[i]!=0:
                    if child1[i] not in cust_child1:
                        cust_child1.append(child1[i])
                    else:
                        child1[i]=cust_miss_child1[j]
                        j+=1

            cust_child2=[]
            cust_repeat_child2=[]
            cust_miss_child2=[]

            for i in range(len(child2)):
                if child2[i]!=0:
                    if child2[i] not in cust_child2:
                        cust_child2.append(child2[i])
                    else:
                        cust_repeat_child2.append(child2[i])

            for  i in list_cust_code:
                if i not in cust_child2:
                    cust_miss_child2.append(i)

            cust_child2=[]
            j=0
            for i in range(len(child2)):
                if  child2[i]!=0:
                    if child2[i] not in cust_child2:
                        cust_child2.append(child2[i])
                    else:
                        child2[i]=cust_miss_child2[j]
                        j+=1

            solution_found1 = Sol(child1)
            solution_found2 = Sol(child2)
            is_sol = all((solution_found1.route_checker(route) for route in solution_found1.routes)) and \
                        all((solution_found2.route_checker(route) for route in solution_found2.routes))

            if is_sol:
                return solution_found1, solution_found2

        if not is_sol:
            if self.verbose > 1:
                print("crossover wasn't able to find a neighbor for this solution")
            return solution1, solution2

        neighbor1 = Sol(child1)
        neighbor2 = Sol(child2)
    
        return neighbor1, neighbor2

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

    def trivial_solution(self, n_cust):
        sol = []
        initial_list = list(range(1,n_cust+1))
        while len(initial_list) > 0:
            sol.append(0)
            sol.append(initial_list.pop(0))
        sol = sol +[0]
        return Sol(sol)

    def random_solution(self, nb_cust) -> VRPTWSolution:
        """
        Generates a random pattern of numbers between 0 and nb_cust, in the form of a solution.
        :param nb_cust: Number of customers wanted in the solution to be generated
        :return: Solution (or nothing)
        """

        def simplify(L, simpleL=None, i=0, on_zero=False):
            if simpleL is None:
                simpleL = []
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

    def set_tracker(self, solution: VRPTWSolution):
        """ Set up for tracker of best solution for a full neighborhood search use """
        self.max_iter = sys.maxsize
        self.best_neighbor = solution
        self.best_cost = solution.cost()

    def track_best(self, solution: VRPTWSolution):
        """ Keeps track of best solution for a full neighborhood search use """
        new_cost = solution.cost()
        if new_cost < self.best_cost:
            self.best_neighbor = solution
            self.best_cost = new_cost
