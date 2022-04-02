import random


def sol_to_list_routes(sol):
    """
    Transforms [0, x1, x2, 0, x3, 0, x4, x5, x6, 0] into [[0, x1, x2, 0], [0, x3, 0], [0, x4, x5, x6, 0]].
    """
    indexes = [i for i, x in enumerate(sol) if x == 0]
    liste_divided = [sol[indexes[i]:indexes[i+1]]+[0] for i in range(len(indexes)-1)]
    return liste_divided


def list_routes_to_sol(sol_list):
    """
    Transforms [[0, x1, x2, 0], [0, x3, 0], [0, x4, x5, x6, 0]] into [0, x1, x2, 0, x3, 0, x4, x5, x6, 0].
    """
    final_sol = []
    for sol in sol_list:
        final_sol += sol[:-1]
    return final_sol + [0]
    

def solution_checker(vrptw, solution, verbose=0):
    """
    Checks whether a solution is legitimate (i.e. meets all necessary constraints) under the context determined
    by a VRPTW instance.
    :param vrptw: VRPTW instance determining the context and rescrictions
    :param solution: Solution to be verified
    :param verbose: Level of verbosity desired
    :return: bool that indicates whether the input 'solution' is a solution or not.
    """
    nb_cust = len(vrptw.customers) # Number of customers (depot included)
    # If all customers are not visited, return False
    if set(solution) != set(range(nb_cust)):
        if verbose >= 1:
            print("All customers are not visited.")
        return False
    # If some nodes (customers) are visited more than once (except for the depot), return False
    nb_depot = solution.count(0)
    if len(solution) != nb_depot+nb_cust-1:
        if verbose >= 1:
            print("There are customers visited more than once.")
        return False

    vehicle = vrptw.vehicle
    volume, weight, cost_km = vehicle.volume, vehicle.weight, vehicle.cost_km 
    sol_routes = sol_to_list_routes(solution)
    time_matrix = vrptw.time_matrix
    customers = vrptw.customers

    for route in sol_routes:
        if verbose >= 2:
            print(f'Working on route: {route}')
        weight_cust, volume_cust = 0, 0
        for identifier in route:
            cust = customers[identifier]
            if verbose >= 3:
                print(cust)
            weight_cust += cust.request_weight
            volume_cust += cust.request_volume
            if verbose >= 2:
                print(f'weight_cust is {weight_cust} and volume_cust is {volume_cust}')
        if verbose >= 2:
            print(weight, volume, weight_cust, volume_cust)
        # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
        if weight < weight_cust or volume < volume_cust :
            if verbose >= 1:
                print(f"The weight (or volume) capacity of the vehicle ({weight}) is < to the total weight asked by customers ({identifier}) on the road ({weight_cust}):")
            return False

        time_delivery = 0
        for index, identifier in enumerate(route[:-1]):
            if verbose >= 2:
                print(f'index={index}, id={identifier}')
            cust = customers[identifier]
            cust_plus_1 = customers[route[index+1]]
            # time_delivery += time_matrix[cust.code_customer,cust_plus_1.code_customer]
            time_delivery += time_matrix[cust.id, cust_plus_1.id]
            # If the vehicle gets there befor the beginning of the customer's time window, return False
            if time_delivery > cust_plus_1.time_window[1]:
                if verbose >= 1:
                    print(f"The vehicle is getting to late ({time_delivery}): customers' ({cust_plus_1.id}) time window's closed {cust_plus_1.time_window[1]}")
                return False
            if time_delivery < cust_plus_1.time_window[0]:
                # waiting for time window to open
                time_delivery = cust_plus_1.time_window[0]
            time_delivery += cust_plus_1.time_service
            # If the end of the delivery is after the end of the customer's time window, return False
            if time_delivery > cust_plus_1.time_window[1]:
                if verbose >= 1:
                    print(f"The vehicle gets there after the end of the time window ({time_delivery} > {cust_plus_1.time_window[1]})")
                return False
    return True


def customers_checker(vrptw, solution, verbose=0):
    """
    Checks whether a solution is legitimate regarding the number of visits of customers under the context determined
    by a VRPTW instance.
    :param vrptw: VRPTW instance determining the context and restrictions
    :param solution: Solution to be verified
    :param verbose: Level of verbosity desired
    :return: bool that indicates whether the input 'solution' does visit all the customers, and if all customers are visited exactly once.
    """
    nb_cust = len(vrptw.customers) # Number of customers (depot included)
    # If all customers are not visited, return False
    if set(solution) != set(range(nb_cust)):
        if verbose >= 1:
            print("All customers are not visited.")
        return False
    # If some nodes (customers) are visited more than once (except for the depot), return False
    nb_depot = solution.count(0)
    if len(solution) != nb_depot+nb_cust-1:
        if verbose >= 1:
            print("There are customers visited more than once.")
        return False
    return True


def route_checker(vrptw, route, verbose=0):
    """
    Checks whether a route is legitimate under the context determined by a VRPTW instance.
    :param vrptw: VRPTW instance determining the context and rescrictions
    :param solution: Solution to be verified
    :param verbose: Level of verbosity desired
    :return: bool that indicates whether the input 'solution' is a solution or not.
    """
    vehicle = vrptw.vehicle
    volume, weight, cost_km = vehicle.volume, vehicle.weight, vehicle.cost_km
    time_matrix = vrptw.time_matrix
    customers = vrptw.customers
    if verbose >= 2:
        print(f'Working on route: {route}')

    weight_cust, volume_cust = 0, 0
    for identifier in route:
        cust = customers[identifier]
        if verbose >= 3:
            print(cust)
        weight_cust += cust.request_weight
        volume_cust += cust.request_volume
        if verbose >= 2:
            print(f'weight_cust is {weight_cust} and volume_cust is {volume_cust}')
    if verbose >= 2:
        print(weight, volume, weight_cust, volume_cust)
    # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
    if weight < weight_cust or volume < volume_cust :
        if verbose >= 1:
            print(f"The weight (or volume) capacity of the vehicle ({weight}) is < to the total weight asked by customers ({identifier}) on the road ({weight_cust}):")
        return False

    time_delivery = 0
    for index, identifier in enumerate(route[:-1]):
        if verbose >= 2:
            print(f'index={index}, id={identifier}')
        cust = customers[identifier]
        cust_plus_1 = customers[route[index+1]]
        # time_delivery += time_matrix[cust.code_customer,cust_plus_1.code_customer]
        time_delivery += time_matrix[cust.id, cust_plus_1.id]
        # If the vehicle gets there befor the beginning of the customer's time window, return False
        if time_delivery > cust_plus_1.time_window[1]:
            if verbose >= 1:
                print(f"The vehicle is getting to late ({time_delivery}): customers' ({cust_plus_1.id}) time window's closed {cust_plus_1.time_window[1]}")
            return False
        if time_delivery < cust_plus_1.time_window[0]:
            # waiting for time window to open
            time_delivery = cust_plus_1.time_window[0]
        time_delivery += cust_plus_1.time_service
        # If the end of the delivery is after the end of the customer's time window, return False
        if time_delivery > cust_plus_1.time_window[1]:
            if verbose >= 1:
                print(f"The vehicle gets there after the end of the time window ({time_delivery} > {cust_plus_1.time_window[1]})")
            return False
    return True


def cost(vrptw, solution, omega=1000, verbose=0):
    """
    returns the total cost of the solution given for the problem given omega is the weight of each vehicle,
    1000 by default.
    """
    # data retrieval
    nb_vehicle = solution.count(0)-1
    distance_matrix = vrptw.distances
    cost_km = vrptw.vehicle.cost_km
    customers = vrptw.customers
    
    # solution given -> list of routes
    sol_list = sol_to_list_routes(solution)
    
    # sum of the distance of each route
    route_length = 0
    for route in sol_list:
        for i in range(len(route)-1):
            route_length += distance_matrix[route[i]][route[i+1]]
    
    # total cost calculation
    total_cost = omega*nb_vehicle + cost_km*route_length
    if verbose >= 1:
        print('Solution:', sol_list)
        print('Total cost of solution:', total_cost)
    return total_cost


def generate_cost_function(vrptw, omega=1000, verbose=0):
    """
    Alternative for creating a cost function that only has to receive a solution (already set with vrptw context).
    returns the set cost function, can be used as follows:
    cost = general_cost_function(vrptw, omega, verbose)
    cost(solution)
    """
    def cost_function(solution):
        return cost(vrptw, solution, omega, verbose)
    return cost_function

