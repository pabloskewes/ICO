import numpy as np
import pandas as pd
import os

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


def load_vehicle(vehicles, MODE="mean", vehicle_nb=None):
    if vehicle_nb:
        volume = vehicles[vehicles["VEHICLE_CODE"]==vehicle_nb]["VEHICLE_TOTAL_VOLUME_M3"]
        weight = vehicles[vehicles["VEHICLE_CODE"]==vehicle_nb]["VEHICLE_TOTAL_WEIGHT_KG"]
        cost_km = vehicles[vehicles["VEHICLE_CODE"]==vehicle_nb]["VEHICLE_VARIABLE_COST_KM"]  
    else :
        volume = getattr(vehicles["VEHICLE_TOTAL_VOLUME_M3"], MODE)()
        weight = getattr(vehicles["VEHICLE_TOTAL_WEIGHT_KG"], MODE)()
        cost_km = getattr(vehicles["VEHICLE_VARIABLE_COST_KM"], MODE)()  
    vehicle = Vehicle(volume, weight, cost_km)
    return vehicle


def load_customers(customers):
    # we supress the lines where the CUSTOMER_CODE repeat itself
    customers = customers.drop_duplicates(subset=["CUSTOMER_CODE"], keep='first')
    # The first customer of the list is the depot, whose id is 0.
    id = 0
    latitude = depots.loc[0,"DEPOT_LATITUDE"]
    longitude = depots.loc[0,"DEPOT_LONGITUDE"]
    time_window = (depots.loc[0,"DEPOT_AVAILABLE_TIME_FROM_MIN"], depots.loc[0,"DEPOT_AVAILABLE_TIME_TO_MIN"])
    request_volume =0
    request_weight = 0
    time_service = 0
    depot = Customer(id, latitude, longitude, time_window, request_volume, request_weight, time_service)
    list_customers = [depot]
    # We add every new customer to the list :
    for i, code in enumerate(customers["CUSTOMER_CODE"], start=1):
        id = i
        latitude = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_LATITUDE"]
        longitude = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_LONGITUDE"]
        time_window = (customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_FROM_MIN"], 
                       customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_TO_MIN"])
        request_volume = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_VOLUME_M3"]
        request_weight = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_WEIGHT_KG"]
        time_service = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"]
        customer = Customer(id, latitude, longitude, time_window, request_volume, request_weight, time_service)
        list_customers.append(customer)
    return list_customers


def data_from_route(path, route_id):
    df = pd.read_excel(path)
    return df[df['ROUTE_ID'] == route_id].drop(['ROUTE_ID'], axis=1)

# ceci est un test pour comprendre les conflits sur git
# REVISAR EL ORDEN DE LOS CUSTOMERS
def matrix_generator(depot_data, customer_data):
    n = len(depot_data)//2
    depot_data.sort_values(['CUSTOMER_CODE']).reset_index(drop=True, inplace=True)
    customer_data.sort_values(['CUSTOMER_CODE_FROM', 'CUSTOMER_CODE_TO']).reset_index(drop=True, inplace=True)
    time_matrix = np.zeros((n+1,n+1))
    distance_matrix = np.zeros((n+1,n+1))
    groups_depot = dict(tuple(depot_data.groupby(['DIRECTION'])))
    time_matrix[0, 1:] = groups_depot['DEPOT->CUSTOMER']['TIME_DISTANCE_MIN'].to_numpy()
    time_matrix[1:, 0] = groups_depot['CUSTOMER->DEPOT']['TIME_DISTANCE_MIN'].to_numpy()
    distance_matrix[0, 1:] = groups_depot['DEPOT->CUSTOMER']['DISTANCE_KM'].to_numpy()
    distance_matrix[1:, 0] = groups_depot['CUSTOMER->DEPOT']['DISTANCE_KM'].to_numpy()
    groups_customer = dict(tuple(customer_data.groupby(['CUSTOMER_CODE_FROM'])))
    keys = np.array(list(groups_customer.keys()))
    for i in range(1, n+1):
        time_matrix[i, 1:] = groups_customer[keys[i-1]]['TIME_DISTANCE_MIN'].to_numpy()
        distance_matrix[i, 1:] = groups_customer[keys[i-1]]['DISTANCE_KM'].to_numpy()
    return time_matrix, distance_matrix, keys