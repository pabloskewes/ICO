import pandas as pd
import numpy as pt

from ICO.codes.loading_models import load_solomon
from solution import *
from loading_models import *

'''
vrptw_test = create_vrptw(CUSTOMER_DIR, DEPOTS_DIR, VEHICLES_DIR, DEPOTS_DISTANCES_DIR, CUSTOMER_DISTANCES_DIR,
                          route_id=2946091, MODE_VEHICLE="mean", vehicle_nb=None)
sol = [*range(len(vrptw_test.customers))]+[0]
nb_cust = len(vrptw_test.customers)
print(sol)
print(nb_cust)
print(set(sol)!= set(range(nb_cust)))
print(set(sol))
print(set(range(nb_cust)))
print(vrptw_test.customers)
print(solution_checker(vrptw_test,sol))
'''

'''
#customers = pd.read_excel(CUSTOMER_DIR)
print(customers.shape)
customers = customers[customers["ROUTE_ID"]==2946091]
print(customers.shape)
customers = customers.drop_duplicates(subset=["CUSTOMER_CODE"], keep='first')
print(customers.shape)
#print(customers["CUSTOMER_CODE"].value_counts)
print(type(customers[customers["CUSTOMER_CODE"]==138157]["CUSTOMER_LATITUDE"]))
print(customers[customers["CUSTOMER_CODE"]==138157]["CUSTOMER_LATITUDE"].shape)
#print(customers[customers["CUSTOMER_CODE"]==138157]["CUSTOMER_LATITUDE"][0])
print(customers[customers["CUSTOMER_CODE"]==138157]["CUSTOMER_LATITUDE"].iloc[0])
#vrptw_solomon = load_solomon("R101.csv")
'''

'''
customers_test = load_customers(customers, depots, route_id=2946091)
print(customers_test[0])
print(customers_test[13])

'''
