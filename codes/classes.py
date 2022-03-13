class Vehicle:
    def __init__(self, volume, weight, cost_km):
        self.volume = volume
        self.weight = weight
        self.cost_km = cost_km

        
class Costumer:
    def __init__(self, x, y, time_window, request_volume, request_weight, time_service):
        self.x = x
        self.y = y
        self.time_window = time_window
        self.request_volume = request_volume
        self.request_weight = request_weight
        self.time_service = time_service
 

class VRPTW:
    '''
    Vehicle Routing Problem Time Windows
    '''
    def __init__(self, costumers, distances, time_matrix, vehicle):
        self.costumers = costumers
        self.distances = distances
        self.time_matrix = time_matrix
        self.vehicle = vehicle
        
        

        




