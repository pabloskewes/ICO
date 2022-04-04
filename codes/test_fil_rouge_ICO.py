import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from loading_models import load_solomon
from vrptw import VRPTW
from solution import VRPTWSolution as Sol

from metaheuristics.simulated_annealing import SimulatedAnnealing

vrptw_context = load_solomon('simple.csv', nb_cust=10, vehicle_speed=100)
vrptw = VRPTW(vrptw_context)
# print(vrptw.solution)
rs = SimulatedAnnealing(t0=30)
# # pprint(vrptw.neighborhood.context.customers)
pprint(vrptw.neighborhood().customers)
# sol = rs.fit_search(vrptw)
# print(sol)
# sol = Sol([0,1,2,3,0,4,5,6,0,7,8,9,10,0])
# print(sol.context)