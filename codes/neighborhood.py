from Test_GVNS_SAV import shaking
from random import randint
from solution import sol_to_list_routes, list_routes_to_sol


class Neighborhood:
    def __init__(self, vrptw):
        self.travel_time = vrptw.time_matrix
        self.demand = [customer.request_weight for customer in vrptw.customers]
        # Only works for solomon instance: only capacity for weight and not for volume.
        self.capacity = vrptw.vehicle.weight
        self.distances = vrptw.distances
        self.service_time = [customer.time_service for customer in vrptw.customers]
        self.ready_time = [customer.time_window[0] for customer in vrptw.customers]
        self.due_time = [customer.time_window[1] for customer in vrptw.customers]
        self.nb_functions = 11

    def get_neighbor(self, solution, function_name=None, verbose=0):
        solution = sol_to_list_routes(solution)
        if function_name is None:
            function_name = randint(0, self.nb_functions - 1)
        if type(function_name) == str:
            names = ['2-opt', 'Or-opt', '2-opstar', 'Relocation', 'Exchange', 'Cross', 'ICross', 'GENI', 'lambda-interchange']
            assert function_name in names, f"Couldn't found a neighborhood function names {function_name}." \
                                           f"\n Options available are {names}."
            dict_names = {names[k]: k+2 for k in range(2, 9)}
            dict_names[names[0]] = 0
            dict_names[names[1]] = 1  # or 2 or 3.
            function_name = dict_names[function_name]
            if verbose >= 1:
                print(f'function name give took value {function_name}')
        assert type(function_name) == int, f'Neighborhood function code must be an integer, got {function_name} instead.'
        k = function_name
        k %= 10
        travel_time, service_time, ready_time, due_time, demand, capacity \
            = self.travel_time, self.service_time, self.ready_time, self.due_time, self.demand, self.capacity
        if k == 0:
            new_sol = shaking(solution, travel_time, service_time, ready_time, due_time, demand, capacity, Neighbor_Str=0)
        elif k == 1:
            new_sol = shaking(solution, travel_time, service_time, ready_time, due_time, demand, capacity, Neighbor_Str=1)
        elif k <= 8:
            new_sol = shaking(solution, travel_time, service_time, ready_time, due_time, demand, capacity, Neighbor_Str=k+2)
        return list_routes_to_sol(new_sol)
    
    def __call__(self, solution, function_name=None, verbose=0):
        new_solution = self.get_neighbor(solution, function_name, verbose)
        return new_solution





