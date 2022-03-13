class Vehicle:
    def __init__(self, volume, weight, cost_km):
        self.volume = volume
        self.weight = weight
        self.cost_km = cost_km

    def __str__(self):
        return f'Vehicle of volume {self.volume}, weight {self.weight}'

    
class Customer:
    def __init__(self, identifier, latitude, longitude, time_window, request_volume, request_weight, time_service):
        self.id = identifier
        self.latitude = latitude
        self.longitude = longitude
        self.time_window = time_window
        self.request_volume = request_volume
        self.request_weight = request_weight
        self.time_service = time_service
        
    def __str__(self):
        return f'This customer\'s id is {self.id}.'


class VRPTW:
    """
    Vehicle Routing Problem Time Windows
    """
    def __init__(self, costumers, distances, time_matrix, vehicle):
        self.costumers = costumers
        self.distances = distances
        self.time_matrix = time_matrix
        self.vehicle = vehicle
        
    def __str__(self):
        return 'Here are the customers : {self.costumers}'
        
        
        




