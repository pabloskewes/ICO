from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np

from metaheuristics.base_problem import Context


@dataclass(frozen=True)
class Vehicle:
    """ Class to store the information of a vehicle """
    vehicle_code: str 
    volume: float
    weight: float
    cost_fixed_per_km: float
    cost_variale_per_km: float
    available_time_from: float
    available_time_to: float

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
    """
    Vehicle Routing Problem Time Windows
    """
    customers: List[Customer]
    distances: np.array
    time_matrix: np.array
    vehicle: Vehicle
    cust_codes: Dict[int, int]





