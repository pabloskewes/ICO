from typing import List, Set, Union, Optional, Dict, Any
from pprint import pformat
import matplotlib.pyplot as plt

from metaheuristics.base_problem import Solution
from context import VRPTWContext

Routes = Union[List[List[int]], List[int]]


def sol_to_list_routes(sol):
    """
    Transforms [0, x1, x2, 0, x3, 0, x4, x5, x6, 0] into [[0, x1, x2, 0], [0, x3, 0], [0, x4, x5, x6, 0]].
    """
    indexes = [i for i, x in enumerate(sol) if x == 0]
    liste_divided = [sol[indexes[i]:indexes[i + 1]] + [0] for i in range(len(indexes) - 1)]
    return liste_divided


def list_routes_to_sol(sol_list):
    """
    Transforms [[0, x1, x2, 0], [0, x3, 0], [0, x4, x5, x6, 0]] into [0, x1, x2, 0, x3, 0, x4, x5, x6, 0].
    """
    final_sol = []
    for sol in sol_list:
        final_sol += sol[:-1]
    return final_sol + [0]


class VRPTWSolution(Solution):
    context: VRPTWContext = None
    omega: int = 500
    static_valid_params = ['omega']

    def __init__(self, routes: Routes = None, params: Optional[Dict] = None):
        super().__init__()

        self.verbose = 0
        if routes is None:
            self.routes = None
            self.sol_code = None
        elif type(routes) == list and type(routes[0]) == int:
            self.routes = sol_to_list_routes(routes)
            self.sol_code = routes
        elif type(routes) == list and type(routes[0]) == list and type(routes[0][0]) == int:
            self.routes = routes
            self.sol_code = list_routes_to_sol(routes)
        else:
            raise Exception('Not a valid form of solution')
        self.set_routes = set(tuple(i) for i in self.routes) if routes is not None else None
        self.graph = []
        if self.sol_code is not None:
            for i in range(1, len(self.sol_code)):
                self.graph.append((self.sol_code[i-1], self.sol_code[i]))

        self.valid_params = ['verbose']
        if params is not None:
            self.set_params(params)

    def __repr__(self):
        return pformat(self.routes)

    def cost(self) -> float:
        """
        returns the total cost of the solution given for the problem given omega is the weight of each vehicle,
        1000 by default.
        """
        # data retrieval
        nb_vehicle = self.sol_code.count(0) - 1
        distance_matrix = self.context.distances
        cost_km = self.context.vehicle.cost_km

        # sum of the distance of each route
        route_length = 0
        for route in self.routes:
            for i in range(len(route) - 1):
                route_length += distance_matrix[route[i], route[i + 1]]

        # total cost calculation
        total_cost = self.omega * nb_vehicle + cost_km * route_length
        if self.verbose >= 2:
            print('Solution:', self.routes)
            print('Total cost of solution:', total_cost)
        return total_cost

    def customers_checker(self) -> bool:
        """
        Checks whether a solution is legitimate regarding the number of visits of customers under the context determined
        by a vrptw context instance.
        :return: bool that indicates whether the input 'solution' does visit all the customers, and if all customers are
        visited exactly once.
        """
        nb_cust = len(self.context.customers)  # Number of customers (depot included)
        # If all customers are not visited, return False
        if set(self.sol_code) != set(range(nb_cust)):
            if self.verbose >= 1:
                print("All customers are not visited.")
            return False
        # If some nodes (customers) are visited more than once (except for the depot), return False
        nb_depot = self.sol_code.count(0)
        if len(self.sol_code) != nb_depot + nb_cust - 1:
            if self.verbose >= 1:
                print("There are customers visited more than once.")
            return False
        return True

    def route_checker(self, route) -> bool:
        """
        Checks whether a route is legitimate under the context determined by a vrptw context instance.
        :param route: Route to check
        :return: bool that indicates whether the input 'solution' is a solution or not.
        """
        vehicle = self.context.vehicle
        volume_capacity, weight_capacity, cost_km = vehicle.volume, vehicle.weight, vehicle.cost_km
        time_matrix = self.context.time_matrix
        customers = self.context.customers
        if self.verbose >= 2:
            print(f'Working on route: {route}')

        weight_cust, volume_cust = 0, 0
        for identifier in route:
            cust = customers[identifier]
            if self.verbose >= 3:
                print(f'Customer = {cust}')
            weight_cust += cust.request_weight
            volume_cust += cust.request_volume
            if self.verbose >= 2:
                print(f'Weight_cust is {weight_cust} and volume_cust is {volume_cust}')
        if self.verbose >= 2:
            print(weight_capacity, volume_capacity, weight_cust, volume_cust)
        # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
        if weight_capacity < weight_cust or volume_capacity < volume_cust:
            if self.verbose >= 1:
                print(
                    f"Weight or volume capacity of the truck exceeded: weight_capacity = {weight_capacity} < weight_cu"
                    f"st = {weight_cust},  volume_capacity = {volume_capacity} < volume_cust = {volume_cust}")
            return False

        time_delivery = 0
        for index, identifier in enumerate(route[:-1]):
            if self.verbose >= 2:
                print(f'index={index}, id={identifier}')
            cust = customers[identifier]
            cust_plus_1 = customers[route[index + 1]]
            # time_delivery += time_matrix[cust.code_customer,cust_plus_1.code_customer]
            time_delivery += time_matrix[cust.id, cust_plus_1.id]
            # If the vehicle gets there befor the beginning of the customer's time window, return False
            if time_delivery > cust_plus_1.time_window[1]:
                if self.verbose >= 1:
                    print(
                        f"The vehicle is getting to late ({time_delivery}): customers' ({cust_plus_1.id}) time window's "
                        f"closed {cust_plus_1.time_window[1]}")
                return False
            if time_delivery < cust_plus_1.time_window[0]:
                # waiting for time window to open
                time_delivery = cust_plus_1.time_window[0]
            time_delivery += cust_plus_1.time_service
            # If the end of the delivery is after the end of the customer's time window, return False
            if time_delivery > cust_plus_1.time_window[1]:
                if self.verbose >= 1:
                    print(
                        f"The vehicle gets there after the end of the time window ({time_delivery} > "
                        f"{cust_plus_1.time_window[1]})")
                return False
        return True

    def checker(self):
        customers_check = self.customers_checker()
        route_check = all((self.route_checker(route) for route in self.routes))
        return customers_check and route_check

    def __eq__(self, other):
        if isinstance(other, VRPTWSolution):
            assert isinstance(other, VRPTWSolution), f"Cannot compare VRPTWSolution type with {other} of type {type(other)}"
            return self.set_routes == other.set_routes
        else:
            return False

    def __le__(self, other):
        assert isinstance(other, VRPTWSolution), f"Cannot compare VRPTWSolution type with {other} of type {type(other)}"
        return len(self.routes) < len(other.routes) or \
               (len(self.routes) == len(other.routes) and self.cost() <= other.cost())

    def __ge__(self, other):
        assert isinstance(other, VRPTWSolution), f"Cannot compare VRPTWSolution type with {other} of type {type(other)}"
        return len(self.routes) > len(other.routes) or \
               (len(self.routes) == len(other.routes) and self.cost() >= other.cost())
    
    def print_graph(self):
        output = ''
        output += str(self.graph[0][0])
        for edge in self.graph:
            output += ' -> ' + str(edge[1])
        print(output)

    def plot_graph(self, name_delta=2, arrows=False):
        customers_list = self.context.customers
        depot = customers_list[0]
        customers = customers_list[1:]
        x_positions = [cust.latitude for cust in customers]
        y_positions = [cust.longitude for cust in customers]

        if arrows: # doesn't work well
            for edge in self.graph:
                cust1, cust2 = customers_list[edge[0]], customers_list[edge[1]]
                x1, y1, x2, y2 = cust1.latitude, cust1.longitude, cust2.latitude, cust2.longitude
                plt.arrow(x=x1, y=y1, dx=x2-x1, dy=y2-y1, width=0.1)
        else:
            for edge in self.graph:
                cust1, cust2 = customers_list[edge[0]], customers_list[edge[1]]
                plt.plot([cust1.latitude, cust2.latitude], [cust1.longitude, cust2.longitude], color='black')
        plt.annotate(text="Depot", xy=(depot.latitude + name_delta, depot.longitude + name_delta), color='red')

        plt.scatter(x=depot.latitude, y=depot.longitude, c='r')
        plt.scatter(x=x_positions, y=y_positions, c='b')

        for cust in customers:
            plt.annotate(text=str(cust.id), xy=(cust.latitude + name_delta, cust.longitude))
        plt.title('Graph representation of solution')
