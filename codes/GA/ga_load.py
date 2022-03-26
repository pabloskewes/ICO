from ga_model import*

def load_customers(customers):
    # we supress the lines where the CUSTOMER_CODE repeat itself
    customers = customers.drop_duplicates(subset=["CUSTOMER_CODE"], keep='first')
    # The first customer of the list is the depot, whose id is 0.
    id = 0
    time_window = (depots.loc[0,"DEPOT_AVAILABLE_TIME_FROM_MIN"], depots.loc[0,"DEPOT_AVAILABLE_TIME_TO_MIN"])
    request_volume =0
    request_weight = 0
    time_service = 0
    depot = Customer(id,time_window, request_volume, request_weight, time_service)
    list_customers = [depot]
    # We add every new customer to the list :
    for i, code in enumerate(customers["CUSTOMER_CODE"], start=1):
        id = i
        time_window = (customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_FROM_MIN"].tolist()[0], 
                       customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_TO_MIN"].tolist()[0])
        request_volume = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_VOLUME_M3"].tolist()[0]
        request_weight = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_WEIGHT_KG"].tolist()[0]
        time_service = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"].tolist()[0]
        customer = Customer(id,time_window, request_volume, request_weight, time_service)
        list_customers.append(customer)
    return list_customers

def load_vehicle(vehicles,vehicle_ids):

    list_vehicles=[]
    for vehicle_id in vehicle_ids:
        volume = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_TOTAL_VOLUME_M3"].tolist()[0]
        weight = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_TOTAL_WEIGHT_KG"].tolist()[0]
        cost_km = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_VARIABLE_COST_KM"].tolist()[0]

        list_vehicles.append(Vehicle(id,volume, weight, cost_km))

    return list_vehicles