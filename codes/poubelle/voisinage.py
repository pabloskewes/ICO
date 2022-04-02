
from Test_GVNS_SAV import *
from solution import *

def vois_two_opt_move(vrptw,sol):
    tour = sol_to_list_routes(sol) # c'est bien ça ?
    distance = vrptw.distances
    travel_time = vrptw.time_matrix
    service_time = [customer.time_service for customer in vrptw.customers]
    ready_time = [customer.time_window[0] for customer in vrptw.customers] # ????? je ne sais pas ce que c'est
    due_time = [customer.time_window[1] for customer in vrptw.customers]
    _,_,_,best_imp = two_opt_move(tour, distance, travel_time, service_time, ready_time, due_time)
    return best_imp


def vois_two_opt_search(vrptw,sol):
    sub_tour = sol_to_list_routes(sol) # c'est bien ça ?
    distance = vrptw.distances
    travel_time = vrptw.time_matrix
    service_time = [customer.time_service for customer in vrptw.customers]
    ready_time = [customer.time_window[0] for customer in vrptw.customers] # ????? je ne sais pas ce que c'est
    due_time = [customer.time_window[1] for customer in vrptw.customers]
    _,_,_,best_imp = two_opt_search(sub_tour, distance, travel_time, service_time, ready_time, due_time)
    return best_imp


# pas fait Or-opt Exchange car je ne comprends pas ce que K est

# pas fait ceux d'après car je ne comprends pas trop ce que tour1 et tour2 sont


