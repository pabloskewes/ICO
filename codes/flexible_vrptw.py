from typing import Optional, Dict
from collections import Counter
import random
from gc import collect

from context import VRPTWContext
from solution import VRPTWSolution, Routes
from neighborhood import VRPTWNeighborhood
from solution_space import VRPTWSolutionSpace
from metaheuristics.base_problem import Problem


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

    def customers_checker(self) -> bool:
        """
        Checks whether a solution is legitimate regarding the number of visits of customers under the context determined
        by a flexible vrptw context instance.
        :return: bool that indicates whether all customers are
        visited exactly once.
        """
        counts = Counter(self.sol_code)
        del counts[0]
        return all(count == 1 for count in counts.values())

    def checker(self):
        customers_check = self.customers_checker()
        route_check = all((self.route_checker(route) for route in self.routes))
        return customers_check and route_check


class FlexVRPTWNeighborhood(VRPTWNeighborhood):
    @classmethod
    def random_solution(cls, nb_cust, force_check_vrptw=None, verbose=0) -> FlexVRPTWSolution:
        tot_customers = len(cls.context.customers) - 1
        if nb_cust < tot_customers // 2:
            return cls.random_solution(nb_cust=tot_customers,
                                       force_check_vrptw=force_check_vrptw,
                                       verbose=verbose)
        numbers = list(range(1, tot_customers + 1))
        random.sample(numbers, nb_cust)
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
                return cls.random_solution(nb_cust=nb_cust - 1,
                                           force_check_vrptw=force_check_vrptw,
                                           verbose=verbose)
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
        solution = FlexVRPTWSolution(code_solution)
        if force_check_vrptw:
            check = solution.checker()
            if not check:
                if verbose >= 1:
                    print('Solution generated is not legitimate, a new one will be created.')
                del numbers, proportion, n_0, zero_positions, zero_pos, zero_pos_candidates
                del solution, string, code_solution
                collect()
                return cls.random_solution(nb_cust=nb_cust - 1,
                                           force_check_vrptw=force_check_vrptw,
                                           verbose=verbose)
            else:
                if verbose >= 1:
                    print(f'A legitimate solution was successfully generated:\n{solution}')
        return solution


class FlexVRPTWSolutionSpace(VRPTWSolutionSpace):
    pass

