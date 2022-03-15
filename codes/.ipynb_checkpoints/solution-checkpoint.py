import numpy as np
import pandas as pd
import os

from classes_ico import Vehicle, Customer, VRPTW
from processing import *

def sol_to_list_routes(sol):
    indexes = [i for i,x in enumerate(sol) if x == 0]
    liste_divided = [sol[indexes[i]:indexes[i+1]]+[0] for i in range(len(indexes)-1)]
    return liste_divided
    

def solution_checker(vrptw, sol):
    nb_cust = len(vrptw.customers) # Number of customers (depot included)
    # If all customers are not visited, return False
    if set(sol)!= set(range(nb_cust)):
        print("Not a solution : all customers are not visited.")
        return False
    # If some nodes (customers) are visited more than once (except for the depot), return False
    nb_depot = sol.count(0)
    if len(sol) != nb_depot+nb_cust-1:
        print("Not a solution : some customers are visited several times.")
        return False
    vehicle = vrptw.vehicle
    volume, weight, cost_km = vehicle.volume, vehicle.weight, vehicle.cost_km 
    sol_routes = sol_to_list_routes(sol)
    time_matrix = vrptw.time_matrix
    customers = vrptw.customers
    for route in sol_routes:
        weight_cust, volume_cust, time_delivery = 0, 0, 0
        for identifier in route:
            cust = [customer for customer in customers if customer.id == identifier][0]
            weight_cust += cust.request_weight
            volume_cust += cust.request_volume
        # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
        if weight < weight_cust or volume < volume_cust :
            print("Not a solution : the weight (or volume) capacity of the vehicle is < to the total weight asked by customers")
            return False
        for index,identifier in enumerate(route):
            cust = [customer for customer in costumers if customer.id == identifier][0]
            cust_plus_1 = [customer for customer in customers if customer.id == route[index+1]][0]
            time_delivery += time_matrix[cust.code_customer,cust_plus_1.code_customer]
            # If the vehicle gets there befor the beginning of the customer's time window, return False
            if time_delivery<cust_plus_1.time_window[0]:
                print("Not a solution : some time windows are not respected")
                return False
            time_delivery += cust_plus_1.time_service
            # If the end of the delivery is after the end of the customer's time window, return False
            if time_delivery>cust_plus_1.time_window[1]:
                print("Not a solution : some time windows are not respected")
                return False
    return True




    
# TEST
#vrptw_test = create_vrptw(CUSTOMER_DIR, DEPOTS_DIR, VEHICLES_DIR, DEPOTS_DISTANCES_DIR, CUSTOMER_DISTANCES_DIR, route_id=2946091, MODE_VEHICLE="mean", vehicle_nb=None)
#print(vrptw_test.customers)
#sol = [range(len(vrptw_test.customers)+1)]+[0]
#solution_checker(vrptw_test, sol)

    

'''
def cost(vrptw, sol):
    w = 1000
    nb_vehicle = 
'''


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
    

        
    
    