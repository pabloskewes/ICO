from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class Vehicle:
    """ Class to store the information of a vehicle """
    volume: float
    weight: float
    cost_km: float


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


@dataclass(frozen=True, repr=False)
class VRPTW:
    """
    Vehicle Routing Problem Time Windows
    """
    customers: List[Customer]
    distances: np.array
    time_matrix: np.array
    vehicle: Vehicle
    cust_codes: int



