from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
from matplotlib.pyplot import scatter

from metaheuristics.base_problem import Context


@dataclass(frozen=True)
class Vehicle:
    """ Class to store the information of a vehicle """
    volume: float
    weight: float
    cost_km: float = None
    vehicle_code: str = None
    cost_fixed_per_km: float = None
    cost_variable_per_km: float = None
    available_time_from: float = None
    available_time_to: float = None


@dataclass(frozen=True)
class Customer:
    """  Class to store the information of a customer """
    id: int
    code_customer: int = field(repr=False)
    latitude: float = field(repr=False)
    longitude: float = field(repr=False)
    time_window: Tuple[float, float]
    request_volume: float
    request_weight: float
    time_service: float


@dataclass(repr=False)
class VRPTWContext(Context):
    """ Vehicle Routing Problem Time Windows data """
    customers: List[Customer]
    distances: np.array
    time_matrix: np.array
    vehicle: Vehicle
    cust_codes: Dict[int, int]

    def show(self):
        customers_list = self.customers
        depot = customers_list[0]
        customers = customers_list[1:]
        vehicle = self.vehicle
        print('Vehicle:')
        print('Volume:', vehicle.volume)
        print('Weight:', vehicle.weight)
        print('Cost KM:', vehicle.cost_km)
        x_positions = [cust.latitude for cust in customers]
        y_positions = [cust.longitude for cust in customers]
        scatter(x=depot.latitude, y=depot.longitude, c='r')
        scatter(x=x_positions, y=y_positions, c='b')
