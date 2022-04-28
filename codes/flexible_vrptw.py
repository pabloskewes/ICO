from typing import Optional, Dict, List
import random
from copy import deepcopy

from context import VRPTWContext
from solution import VRPTWSolution, Routes, Solution
from neighborhood import VRPTWNeighborhood
from solution_space import VRPTWSolutionSpace


class FlexVRPTWSolution(VRPTWSolution):
    context: VRPTWContext = None
    omega: int = 500
    cust_penalty: int = 300
    static_valid_params = ['omega', 'cust_penalty']

    def __init__(self, routes: Routes = None, params: Optional[Dict] = None):
        super().__init__(routes=routes, params=params)

        self.tot_customers = len(self.context.customers)
        self.missing_customers = set(range(self.tot_customers)) - set(self.sol_code)

    def cost(self) -> float:
        """
        returns the total cost of the solution given for the problem given omega is the weight of each vehicle,
        1000 by default.
        """
        total_cost = super().cost()
        total_cost += len(self.missing_customers) * self.cust_penalty
        return total_cost

    def checker(self):
        not_repeated_customers_check = self.not_repeated_customers_checker()
        route_check = all((self.route_checker(route) for route in self.routes))
        return not_repeated_customers_check and route_check


class FlexVRPTWNeighborhood(VRPTWNeighborhood):
    context: VRPTWContext = None

    # NEIGHBORHOOD FUNCTION 1 - INTRA ROUTE SWAP
    def intra_route_swap(self, solution: FlexVRPTWSolution) -> FlexVRPTWSolution:
        new_sol = super().intra_route_swap(solution)
        return FlexVRPTWSolution(new_sol.sol_code)

    # NEIGHBORHOOD FUNCTION 9 - SKIP CUSTOMER
    def skip_customer(self, solution) -> Solution:
        """ Moves a random client to another position on its same route """
        if self.full_search: self.set_tracker(solution)
        routes = solution.routes
        new_routes = deepcopy(routes)
        s = VRPTWSolution()
        is_sol = False
        available_routes_index = list(range(len(routes)))
        routes_iter = 0
        while not is_sol and routes_iter < self.max_iter and available_routes_index:
            r_index = random.choice(available_routes_index)
            available_routes_index.remove(r_index)

            route: List[int] = routes[r_index].copy()
            if len(route) < 4:
                continue
            cust_iter = 0
            available_cust_index = list(range(1, len(route)-1))
            while not is_sol and cust_iter < self.max_iter and available_cust_index:
                index_cust = random.choice(available_cust_index)
                available_cust_index.remove(index_cust)
                # removing the customer from the route
                route.remove(index_cust)
                is_sol = s.route_checker(route)
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

    def random_solution(self, nb_cust) -> FlexVRPTWSolution:
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
        n_iter = 0
        min_length = len(numbers)//2
        solution = None
        
        while not is_sol:
            n_iter += 1
            if (n_iter % self.max_iter == 0) and len(numbers) > min_length:
                r = random.randint(0, len(numbers))
                numbers.remove(numbers[r])
            
            random.shuffle(numbers)
            proportion = random.choice([0.05, 0.1, 0.15, 0.2])
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
            solution = FlexVRPTWSolution(code_solution)
       
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


class FlexVRPTWSolutionSpace(VRPTWSolutionSpace):
    context: VRPTWContext = None
    pass

