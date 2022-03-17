from Test_GVNS_SAV import shaking
from random import randint
import random
from solution import sol_to_list_routes, list_routes_to_sol, solution_checker, random_solution


class Neighborhood:
    def __init__(self, vrptw):
        """
        Initializes a solution neighborhood manager of a VRPTW.
        :param vrptw: VRPTW instance to be used as context.
        """
        self.vrptw = vrptw
        self.travel_time = vrptw.time_matrix
        self.demand = [customer.request_weight for customer in vrptw.customers]
        # Only works for solomon instance: only capacity for weight and not for volume.
        self.capacity = vrptw.vehicle.weight
        self.distances = vrptw.distances
        self.service_time = [customer.time_service for customer in vrptw.customers]
        self.ready_time = [customer.time_window[0] for customer in vrptw.customers]
        self.due_time = [customer.time_window[1] for customer in vrptw.customers]
        self.nb_functions = 11
        self.working_functions = [0, 1]

    def get_neighbor(self, solution, function_name=None, force_check=False, force_working_functions=True, verbose=0):
        """
        Neighbor function to find a neighboring solution.
        :param solution: Solution to use
        :param function_name: Name of the neighborhood function to be used, it can be a code from 0 to 8
        or its name directly, e.g.: '2-opt'.
        :param force_check: If true, then the solution found is checked for legitimacy using the 'solution_checker'
        function of the solution.py module.
        :param force_working_functions: f true, then it is imposed that the neighborhood function to use is one that
        has no errors (predefined codes in the 'self.working_functions' field).
        :param verbose: Level of verbosity desired
        :return: New solution belonging to the neighborhood of the solution given in the input. If 'force_check'
        is True and it also proves that the given solution is not legitimate, then this method will return an
        empty solution: new_sol = [].
        """
        solution = sol_to_list_routes(solution)
        if function_name is None:
            function_name = randint(0, self.nb_functions - 1)
        names = ['2-opt', 'Or-opt', '2-opstar', 'Relocation', 'Exchange', 'Cross', 'ICross', 'GENI',
                 'lambda-interchange']
        dict_names = {names[k]: k + 2 for k in range(2, 9)}
        dict_names[names[0]] = 0
        dict_names[names[1]] = 1  # or 2 or 3.
        if type(function_name) == str:
            assert function_name in names, f"Couldn't found a neighborhood function names {function_name}." \
                                           f"\n Options available are {names}."
            k = dict_names[function_name]
            if verbose >= 1:
                print(f'function name give took value {function_name}')
        else:  # if function_name is not a string, we assume it's the code directly (int between 0 and 8).
            k = function_name
        assert type(k) == int, f'Neighborhood function code must be an integer, got {k} instead.'
        k %= 9
        travel_time, service_time, ready_time, due_time, demand, capacity \
            = self.travel_time, self.service_time, self.ready_time, self.due_time, self.demand, self.capacity
        assert k in range(0, 8+1), f'k must be in [0, 8], got {k} instead'
        if force_working_functions:
            if verbose >= 1:
                print('forcing use of working neighborhood function')
            if k not in self.working_functions:
                if verbose >= 1:
                    print(f'function with code {k} requested but could not be used, random available function will be used instead')
                k = random.choice(self.working_functions)
            else:
                if verbose >= 1:
                    print(f'function with code {k} requested and available')
        if verbose >= 1:
            if k in [2, 3]:
                k = 1
            key_list = list(dict_names.keys())
            val_list = list(dict_names.values())
            position = val_list.index(k)
            key = key_list[position]
            if verbose >= 3:
                print(f'key list: {key_list}')
                print(f'val list: {val_list}')
                print(f'pos:{position}')
                print(f'key:{key}')
            print(f'k took value {k}: apply operation named {key}')
        if k == 0:
            new_sol = shaking(solution, travel_time, service_time, ready_time, due_time, demand, Neighbor_Str=0)
        elif k == 1:
            new_sol = shaking(solution, travel_time, service_time, ready_time, due_time, demand, Neighbor_Str=1)
        else:  # k <= 8
            new_sol = shaking(solution, travel_time, service_time, ready_time, due_time, demand, Neighbor_Str=k+2)
        new_sol = list_routes_to_sol(new_sol)
        if force_check:
            if verbose >= 1:
                print(f'force-checking if solution found ({new_sol}) is legitime in this context')
            check = solution_checker(vrptw=self.vrptw, solution=new_sol, verbose=verbose)
            if check:
                if verbose >= 1:
                    print(f'solution {new_sol} is legitime !')
            else:
                if verbose >= 1:
                    print(f'solution found is NOT a legitime solution, neighborhood function will return an empty list')
                new_sol = []
        return new_sol
    
    def __call__(self, solution, function_name=None, force_check=False, force_working_functions=True, verbose=0):
        new_solution = self.get_neighbor(solution=solution,
                                         function_name=function_name,
                                         force_check=force_check,
                                         force_working_functions=force_working_functions,
                                         verbose=verbose)
        return new_solution

    def shuffle(self, solution=None, verbose=0):
        """
        Apply the random_solution function to obtain a valid random solution (you can't even say it's really in the
        neighborhood of the solution, but it can be very useful).
        :param solution: Nothing, only for aesthetic purposes.
        :param verbose: Level of verbosity desired.
        :return:
        """
        r_sol = random_solution(nb_cust=len(self.vrptw.customers)-1, force_check_vrptw=self.vrptw, verbose=verbose)
        return r_sol




