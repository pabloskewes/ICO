import random

from metaheuristics.problem import Solution
from types import List, Set, Union
from vrptw import VRPTW
Routes = Union[List[List[int]], List[int]]


class VRPTW_Solution(Solution):
    OMEGA = 1000

    def __init__(self, routes: Routes, vrptw):
        self.context =
        pass

    def cost(self):
        """
        returns the total cost of the solution given for the problem given omega is the weight of each vehicle,
        1000 by default.
        """
        # data retrieval
        nb_vehicle = solution.count(0) - 1
        distance_matrix = vrptw.distances
        cost_km = vrptw.vehicle.cost_km
        customers = vrptw.customers

        # solution given -> list of routes
        sol_list = sol_to_list_routes(solution)

        # sum of the distance of each route
        route_length = 0
        for route in sol_list:
            for i in range(len(route) - 1):
                route_length += distance_matrix[route[i]][route[i + 1]]

        # total cost calculation
        total_cost = omega * nb_vehicle + cost_km * route_length
        if verbose >= 1:
            print('Solution:', sol_list)
            print('Total cost of solution:', total_cost)
        return total_cost






