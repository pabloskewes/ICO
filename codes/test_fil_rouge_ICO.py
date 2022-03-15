import pandas as pd
import numpy as pt
from solution import *
from processing import *

vrptw_test = create_vrptw(CUSTOMER_DIR, DEPOTS_DIR, VEHICLES_DIR, DEPOTS_DISTANCES_DIR, CUSTOMER_DISTANCES_DIR,
                          route_id=2946091, MODE_VEHICLE="mean", vehicle_nb=None)
sol = [*range(len(vrptw_test.customers))]+[0]
nb_cust = len(vrptw_test.customers)
print(sol)
print(nb_cust)
print(set(sol)!= set(range(nb_cust)))
print(set(sol))
print(set(range(nb_cust)))
print(vrptw_test.customers[-1])
print(solution_checker(vrptw_test,sol))



