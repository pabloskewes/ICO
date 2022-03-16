import numpy as np
import pandas as pd
import os
from math import sqrt
import sys

from classes_ico import Vehicle, Customer, VRPTW

ROOT_DIR = os.path.abspath('..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CUSTOMER_DIR = os.path.join(DATA_DIR, '2_detail_table_customers.xls')
VEHICLES_DIR = os.path.join(DATA_DIR, '3_detail_table_vehicles.xls')
DEPOTS_DIR = os.path.join(DATA_DIR, '4_detail_table_depots.xls')
CONSTRAINTS_DIR = os.path.join(DATA_DIR, '5_detail_table_constraints_sdvrp.xls')
DEPOTS_DISTANCES_DIR = os.path.join(DATA_DIR, '6_detail_table_cust_depots_distances.xls')
CUSTOMER_DISTANCES_DIR = os.path.join(DATA_DIR, '7_detail_table_cust_cust_distances.xls')

customers = pd.read_excel(CUSTOMER_DIR)
vehicles = pd.read_excel(VEHICLES_DIR)
depots = pd.read_excel(DEPOTS_DIR)
constraints = pd.read_excel(CONSTRAINTS_DIR)
depots_dist = pd.read_excel(DEPOTS_DISTANCES_DIR)
customers_dist = pd.read_excel(CUSTOMER_DISTANCES_DIR)


def load_vehicle(vehicles, MODE_VEHICLE="mean", vehicle_nb=None):
    if vehicle_nb:
        volume = vehicles[vehicles["VEHICLE_CODE"]==vehicle_nb]["VEHICLE_TOTAL_VOLUME_M3"]
        weight = vehicles[vehicles["VEHICLE_CODE"]==vehicle_nb]["VEHICLE_TOTAL_WEIGHT_KG"]
        cost_km = vehicles[vehicles["VEHICLE_CODE"]==vehicle_nb]["VEHICLE_VARIABLE_COST_KM"]  
    else :
        volume = getattr(vehicles["VEHICLE_TOTAL_VOLUME_M3"], MODE_VEHICLE)()
        weight = getattr(vehicles["VEHICLE_TOTAL_WEIGHT_KG"], MODE_VEHICLE)()
        cost_km = getattr(vehicles["VEHICLE_VARIABLE_COST_KM"], MODE_VEHICLE)()  
    vehicle = Vehicle(volume, weight, cost_km)
    return vehicle


def load_customers(customers, depots, route_id=2946091):
    customers = customers[customers["ROUTE_ID"]==route_id]
    # we supress the lines where the CUSTOMER_CODE repeat itself
    customers = customers.drop_duplicates(subset=["CUSTOMER_CODE"], keep='first')
    # The first customer of the list is the depot, whose id is 0.
    identifier = 0
    customer_code = 1000
    latitude = depots.loc[0,"DEPOT_LATITUDE"]
    longitude = depots.loc[0,"DEPOT_LONGITUDE"]
    #time_window = (depots.loc[0,"DEPOT_AVAILABLE_TIME_FROM_MIN"], depots.loc[0,"DEPOT_AVAILABLE_TIME_TO_MIN"])
    time_window = (0,1440)
    request_volume =0
    request_weight = 0
    time_service = 0
    depot = Customer(identifier, customer_code, latitude, longitude, time_window, request_volume, request_weight, time_service)
    list_customers = [depot]
    # We add every new customer to the list :
    for i, code in enumerate(customers["CUSTOMER_CODE"], start=1):
        identifier = i
        customer_code = code
        latitude = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_LATITUDE"].iloc[0]
        longitude = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_LONGITUDE"].iloc[0]
        time_window = (customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_FROM_MIN"].iloc[0],
                       customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_TO_MIN"].iloc[0])
        request_volume = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_VOLUME_M3"].iloc[0]
        request_weight = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_WEIGHT_KG"].iloc[0]
        time_service = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"].iloc[0]
        customer = Customer(identifier, customer_code, latitude, longitude, 
                            time_window, request_volume, request_weight, time_service)
        list_customers.append(customer)
    return list_customers


def data_from_route(path, route_id):
    df = pd.read_excel(path)
    return df[df['ROUTE_ID'] == route_id].drop(['ROUTE_ID'], axis=1)

# matrix_generator
# Creates the time and distances matrix, and also returns the list with the customer codes used
def matrix_generator(depot_data, customer_data, route_id):
    
    # data filtering and reordering
    depot_data = depot_data[depot_data['ROUTE_ID'] == route_id].drop(['ROUTE_ID'], axis=1)
    customer_data = customer_data[customer_data['ROUTE_ID'] == route_id].drop(['ROUTE_ID'], axis=1)
    depot_data['CUSTOMER_CODE'] = pd.to_numeric(depot_data['CUSTOMER_CODE'], errors='ignore', downcast='integer')
    customer_data['CUSTOMER_CODE_FROM'] = pd.to_numeric(customer_data['CUSTOMER_CODE_FROM'], errors='ignore', downcast='integer')
    customer_data['CUSTOMER_CODE_TO'] = pd.to_numeric(customer_data['CUSTOMER_CODE_TO'], downcast='integer')
    depot_data = depot_data.sort_values(['CUSTOMER_CODE']).reset_index(drop=True)
    customer_data = customer_data.sort_values(['CUSTOMER_CODE_FROM', 'CUSTOMER_CODE_TO']).reset_index(drop=True)
    
    # matrix creation and filling
    n = len(depot_data)//2
    time_matrix = np.zeros((n+1,n+1))
    distance_matrix = np.zeros((n+1,n+1))
    groups_depot = dict(tuple(depot_data.groupby(['DIRECTION'])))
    groups_customer = dict(tuple(customer_data.groupby(['CUSTOMER_CODE_FROM'])))
    keys = np.array(list(groups_customer.keys()))
    
    time_matrix[0, 1:] = groups_depot['DEPOT->CUSTOMER']['TIME_DISTANCE_MIN'].to_numpy()
    time_matrix[1:, 0] = groups_depot['CUSTOMER->DEPOT']['TIME_DISTANCE_MIN'].to_numpy()
    distance_matrix[0, 1:] = groups_depot['DEPOT->CUSTOMER']['DISTANCE_KM'].to_numpy()
    distance_matrix[1:, 0] = groups_depot['CUSTOMER->DEPOT']['DISTANCE_KM'].to_numpy()   
    for i in range(1, n+1):
        time_matrix[i, 1:] = groups_customer[keys[i-1]]['TIME_DISTANCE_MIN'].to_numpy()
        distance_matrix[i, 1:] = groups_customer[keys[i-1]]['DISTANCE_KM'].to_numpy()
    
    return time_matrix, distance_matrix, keys


# Create VRPTW :
def create_vrptw(CUSTOMER_DIR, DEPOTS_DIR, VEHICLES_DIR, DEPOTS_DISTANCES_DIR, CUSTOMER_DISTANCES_DIR, route_id=2946091, MODE_VEHICLE="mean", vehicle_nb=None):
    customers = pd.read_excel(CUSTOMER_DIR)
    vehicles = pd.read_excel(VEHICLES_DIR)
    depots = pd.read_excel(DEPOTS_DIR)
    depots_dist = pd.read_excel(DEPOTS_DISTANCES_DIR)
    customers_dist = pd.read_excel(CUSTOMER_DISTANCES_DIR)
    list_costumers = load_customers(customers, depots, route_id=2946091)
    time_matrix, distances, cust_codes = matrix_generator(depots_dist, customers_dist, route_id)
    vehicle = load_vehicle(vehicles, MODE_VEHICLE="mean", vehicle_nb=None)
    vrptw = VRPTW(customers, distances, time_matrix, vehicle, cust_codes)
    return vrptw

#vrptw_test = create_vrptw(CUSTOMER_DIR, DEPOTS_DIR, VEHICLES_DIR, DEPOTS_DISTANCES_DIR, CUSTOMER_DISTANCES_DIR, route_id=2946091, MODE_VEHICLE="mean", vehicle_nb=None)

#print(vrptw_test)



def load_solomon(filename):
    ROOT_DIR = os.path.abspath('..')
    DATA_DIR = os.path.join(ROOT_DIR, 'data_solomon')
    DIR = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(DIR)
    vehicle = Vehicle(volume=sys.maxsize,
                      weight=df.at[0, 'CAPACITY'],
                      cost_km=1)
    df = df.drop('CAPACITY', axis=1)
    n = len(df)
    customers = []
    for k in range(n):
        cust = Customer(identifier=k,
                 code_customer=k,
                 latitude=df.at[k,'XCOORD'],
                 longitude=df.at[k,'YCOORD'],
                 time_window=(df.at[k,'READYTIME'], df.at[k, 'DUETIME']),
                 request_volume=0,
                 request_weight=df.at[k,'DEMAND'],
                 time_service=df.at[k,'SERVICETIME'])
        customers.append(cust)
    cust_codes = {i:i for i in range(n)}
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            x1, y1 = df.at[i, 'XCOORD'], df.at[i, 'YCOORD']
            x2, y2 = df.at[j, 'XCOORD'], df.at[j, 'YCOORD']
            distances[i,j] = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    vrptw = VRPTW(customers=customers,
                  distances=distances,
                  time_matrix=distances,
                  vehicle=vehicle,
                  cust_codes=cust_codes)
    return vrptw


#if __name__ == '__main__':
    
