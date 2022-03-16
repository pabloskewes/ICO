def sol_to_list_routes(sol):
    indexes = [i for i,x in enumerate(sol) if x == 0]
    liste_divided = [sol[indexes[i]:indexes[i+1]]+[0] for i in range(len(indexes)-1)]
    return liste_divided


def list_routes_to_sol(sol_list):
    final_sol = []
    for sol in sol_list:
        final_sol += sol[:-1]
    return final_sol + [0]
    

def solution_checker(vrptw, sol):
    nb_cust = len(vrptw.customers) # Number of customers (depot included)
    # If all customers are not visited, return False
    if set(sol)!= set(range(nb_cust)):
        print("All customers are not visited.")
        return False
    # If some nodes (customers) are visited more than once (except for the depot), return False
    nb_depot = sol.count(0)
    if len(sol) != nb_depot+nb_cust-1:
        print("There are customers visited more than once.")
        return False
    vehicle = vrptw.vehicle
    volume, weight, cost_km = vehicle.volume, vehicle.weight, vehicle.cost_km 
    sol_routes = sol_to_list_routes(sol)
    time_matrix = vrptw.time_matrix
    customers = vrptw.customers
    print("")
    for route in sol_routes:
        weight_cust, volume_cust, time_delivery = 0, 0, 0
        for identifier in route:
            cust = customers[identifier]
            print(cust)
            weight_cust += cust.request_weight
            volume_cust += cust.request_volume
            print(f'weight_cust is {weight_cust} and volume_cust is {volume_cust}')
        # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
        print(weight, volume, weight_cust, volume_cust)
        if weight < weight_cust or volume < volume_cust :
            print("The weight (or volume) capacity of the vehicle is < to the total weight asked by customers on the road :", route)
            return False
        for index,identifier in enumerate(route):
            cust = customers[identifier]
            cust_plus_1 = customers[route[index+1]]
            #time_delivery += time_matrix[cust.code_customer,cust_plus_1.code_customer]
            time_delivery += time_matrix[cust.id,cust_plus_1.id]
            # If the vehicle gets there befor the beginning of the customer's time window, return False
            if time_delivery<cust_plus_1.time_window[0]:
                time_delivery=cust_plus_1.time_window[0]
            time_delivery += cust_plus_1.time_service
            # If the end of the delivery is after the end of the customer's time window, return False
            if time_delivery>cust_plus_1.time_window[1]:
                print("The vehicle gets there after the end of the time window")
                return False
    return True
    
    
# cost
# returns the total cost of the solution given for the problem given
# omega is the weight of each vehicle, 1000 by default
def cost(vrptw, sol, omega=1000):
    # data retrieval
    nb_vehicle = sol.count(0)-1
    distance_matrix = vrptw.distances
    cost_km = vrptw.vehicle.cost_km
    customers = vrptw.customers
    
    # solution given -> list of routes
    sol_list = sol_to_list_routes(sol)
    
    # sum of the distance of each route
    route_length = 0
    for route in sol_list:
        for i in range(len(route)-1):
            route_length += distance_matrix[route[i]][route[i+1]]
    
    # total cost calculation
    total_cost = omega*nb_vehicle + cost_km*route_length
    print('Solution:', sol_list)
    print('Total cost of solution:', total_cost)
    return total_cost


'''
A JETER PLUS TARD :
    for route in sol_routes:
        weight_cust, volume_cust, time_delivery = 0, 0, 0
        for cust in vrptw.customers:
            weight_cust += cust.request_weight
            volume_cust += cust.request_volume
        # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
        if weight < weight_cust or volume < volume_cust :
            return False
        for i in range(len(vrptw.customers)):
            time_delivery += time_matrix[vrptw.customers[i].code_customer,vrptw.customers[i+1].code_customer]
            # If the vehicle gets there befor the beginning of the customer's time window, return False
            if time_delivery<vrptw.customers[i+1].time_window[0]:
                return False
            time_delivery += vrptw.customers[i+1].time_service
            # If the end of the delivery is after the end of the customer's time window, return False
            if time_delivery>vrptw.customers[i+1].time_window[1]:
                return False
    return True
'''
    

        
    
    