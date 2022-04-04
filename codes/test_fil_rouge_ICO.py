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
init_sol = Sol([[0, 7, 5, 0], [0, 6, 0], [0, 8, 2, 0], [0, 3, 4, 9, 10, 1, 0]])
rs = SimulatedAnnealing(t0=30, neighborhood_params={'verbose': 0, 'init_sol': 'random'})

print('init sol:')
print(init_sol)
print('cost =', init_sol.cost())
sol = rs.fit_search(vrptw)
print('sol found:')
print(sol)
print('cost =', sol.cost())
