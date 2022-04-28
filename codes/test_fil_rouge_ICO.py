import threading
import multiprocessing as mp
import time
import random
from metaheuristics.simulated_annealing import SimulatedAnnealing
from loading_models import load_solomon
from vrptw import VRPTW


vrptw_data = load_solomon('A50.csv', nb_cust=None, vehicle_speed=100)
vrptw = VRPTW(vrptw_data)

neighborhood_params = {'verbose': 0,
                       'init_sol': 'random',
                       'choose_mode': 'random',
                       'use_methods': [1, 2, 3, 4, 5, 6, 7, 8],
                       'force_new_sol': True,
                       'full_search': True}

# rs = SimulatedAnnealing(t0=30, progress_bar=True, neighborhood_params=neighborhood_params)
# rs_sol = rs.fit_search(vrptw)
# rs.plot_evolution_cost(figsize=(17,5))

print("Number of processors: ", mp.cpu_count())

# # Python Threading Example for Beginners
# # First Method
# def greet_them(people):
#     for person in people:
#         print("Hello Dear " + person + ". How are you?")
#         time.sleep(0.5)
#
#
# # Second Method
# def assign_id(people):
#     i = 1
#     for person in people:
#         print("Hey! {}, your id is {}.".format(person, i))
#         i += 1
#         time.sleep(0.5)
#
#
# people = ['Richard', 'Dinesh', 'Elrich', 'Gilfoyle', 'Gevin']
#
# t = time.time()
#
# # Created the Threads
# t1 = threading.Thread(target=greet_them, args=(people,))
# t2 = threading.Thread(target=assign_id, args=(people,))
#
# # Started the threads
# t1.start()
# t2.start()
#
# # Joined the threads
# t1.join()
# t2.join()
#
# print("Woaahh!! My work is finished..")
# print("I took " + str(time.time() - t))
