class Vehicle:
    def __init__(self, volume, weight, cost_km):
        self.volume = volume
        self.weight = weight
        self.cost_km = cost_km

    def __str__(self):
        return f'Vehicle of volume {self.volume}, weight {self.weight}'

    
class Customer:
    def __init__(self, identifier, code_customer, latitude, longitude, time_window, request_volume, request_weight, time_service):
        self.id = identifier
        self.code_customer = code_customer
        self.latitude = latitude
        self.longitude = longitude
        self.time_window = time_window
        self.request_volume = request_volume
        self.request_weight = request_weight
        self.time_service = time_service
        
    def __str__(self):
        return f'This customer\'s id is {self.id}, its code_customer is {self.code_customer}, ' \
               f'its latitude is {self.latitude}, its longitude is {self.longitude}, ' \
               f'its time window is {self.time_window}, its volume requested is {self.request_volume},' \
               f'its weight requested is {self.request_weight}, its time service is {self.time_service}.'


class VRPTW:
    """
    Vehicle Routing Problem Time Windows
    """
    def __init__(self, customers, distances, time_matrix, vehicle, cust_codes):
        self.customers = customers
        self.distances = distances
        self.time_matrix = time_matrix
        self.vehicle = vehicle

    def __str__(self):
        return f'Here are the customers : {self.customers}'
        
        
        




