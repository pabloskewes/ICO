import random


def random_solution(nb_cust, force_check_vrptw=None, verbose=0):
    """
    Generates a random pattern of numbers between 0 and nb_cust, in the form of a solution.
    :param nb_cust: Number of customers wanted in the solution to be generated
    :param force_check_vrptw: The default is None and does nothing. When delivering a VRPTW instance in this parameter,
    the legitimacy of the generated solution will be checked (using 'check_solution') based on the context of
    that particular VRPTW instance.
    :param verbose: Level of verbosity desired
    :return: Solution (or nothing)
    """
    numbers = list(range(1, nb_cust+1))
    random.shuffle(numbers)
    proportion = random.choice([0.2, 0.3, 0.4])
    n_0 = int(nb_cust*proportion)
    zero_positions = []
    zero_pos_candidates = list(range(1, nb_cust-1))
    for _ in range(n_0):
        if verbose >= 2:
            print('candidates:', zero_pos_candidates)
        try:
            zero_pos = random.choice(zero_pos_candidates)
        except IndexError:
            if verbose >= 1:
                print('A problem ocurred, generating new random solution')
            return random_solution(nb_cust=nb_cust, force_check_vrptw=force_check_vrptw)
        if verbose >= 2:
            print('zero_pos chosen:', zero_pos)
        zero_pos_candidates = list(set(zero_pos_candidates) - {zero_pos, zero_pos+1, zero_pos-1})
        zero_positions.append(zero_pos)
    for pos in zero_positions:
        numbers.insert(pos, 0)
    solution = [0] + numbers + [0]
    string = str(solution).replace('0, 0, 0', '0').replace('0, 0', '0')
    solution = list(map(int, string.strip('][').split(',')))
    if force_check_vrptw:
        check = solution_checker(vrptw=force_check_vrptw, solution=solution)
        if not check:
            if verbose >= 1:
                print('Solution generated is not legitimate, a new one will be created.')
            return random_solution(nb_cust=nb_cust, force_check_vrptw=force_check_vrptw)
        else:
            if verbose >= 1:
                print(f'A legitimate solution was successfully generated:\n{solution}')
    return solution

    # def numbers_close(number_list):
    #     diff = set()
    #     n = len(number_list)
    #     for i in range(n):
    #         for j in range(i+1, n):
    #             difference = abs(number_list[i] - number_list[j])
    #             diff = diff.union(difference)
    #     return {1, 0, -1}.issubset(diff)
    #
    # while True:
    #     zero_positions = random.sample(range(2, nb_cust-1))
    #     if not numbers_close()


def sol_to_list_routes(sol):
    indexes = [i for i, x in enumerate(sol) if x == 0]
    liste_divided = [sol[indexes[i]:indexes[i+1]]+[0] for i in range(len(indexes)-1)]
    return liste_divided


def list_routes_to_sol(sol_list):
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
    if set(solution)!= set(range(nb_cust)):
        print("All customers are not visited.")
        return False
    # If some nodes (customers) are visited more than once (except for the depot), return False
    nb_depot = solution.count(0)
    if len(solution) != nb_depot+nb_cust-1:
        print("There are customers visited more than once.")
        return False
    vehicle = vrptw.vehicle
    volume, weight, cost_km = vehicle.volume, vehicle.weight, vehicle.cost_km 
    sol_routes = sol_to_list_routes(solution)
    time_matrix = vrptw.time_matrix
    customers = vrptw.customers
    for route in sol_routes:
        if verbose >= 1:
            print(f'Working on route: {route}')
        weight_cust, volume_cust = 0, 0
        for identifier in route:
            cust = customers[identifier]
            if verbose >= 2:
                print(cust)
            weight_cust += cust.request_weight
            volume_cust += cust.request_volume
            print(f'weight_cust is {weight_cust} and volume_cust is {volume_cust}')
        # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
        if verbose >= 1:
            print(weight, volume, weight_cust, volume_cust)
        if weight < weight_cust or volume < volume_cust :
            print(f"The weight (or volume) capacity of the vehicle ({weight}) is < to the total weight asked by customers ({identifier}) on the road ({weight_cust}):")
            return False
        time_delivery = 0
        for index, identifier in enumerate(route[:-1]):
            print(f'index={index}, id={identifier}')
            cust = customers[identifier]
            cust_plus_1 = customers[route[index+1]]
            # time_delivery += time_matrix[cust.code_customer,cust_plus_1.code_customer]
            time_delivery += time_matrix[cust.id, cust_plus_1.id]
            # If the vehicle gets there befor the beginning of the customer's time window, return False
            if time_delivery > cust_plus_1.time_window[1]:
                print(f"The vehicle is getting to late ({time_delivery}): customers' ({cust_plus_1.id}) time window's closed {cust_plus_1.time_window[1]}")
                return False
            if time_delivery < cust_plus_1.time_window[0]:
                # waiting for time window to open
                time_delivery = cust_plus_1.time_window[0]
            time_delivery += cust_plus_1.time_service
            # If the end of the delivery is after the end of the customer's time window, return False
            if time_delivery > cust_plus_1.time_window[1]:
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
    Alternative for creating a cost function that only has to receive a solution (already set with vrptw context.
    returns the set cost function, can be used as follows:
    cost = general_cost_function(vrptw, omega, verbose)
    cost(solution)
    """
    def cost_function(solution):
        return cost(vrptw, solution, omega, verbose)
    return cost_function
