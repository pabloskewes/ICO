import random
import statistics
import pandas as pd
import os
import sys 
from math import sqrt
import copy
import numpy as np

'''
Category:
Identifier      Content
--------------------------------------------------------
FF              code bloques for the data from Fil rouge
SS              code bloques for the SMA
CC              code bloques of definition of classes
DD              code bloques for the simple data
'''


def sol_to_list_routes(sol):
    """
    Transforms [0, x1, x2, 0, x3, 0, x4, x5, x6, 0] into [[0, x1, x2, 0], [0, x3, 0], [0, x4, x5, x6, 0]].
    """
    indexes = [i for i, x in enumerate(sol) if x == 0]
    liste_divided = [sol[indexes[i]:indexes[i+1]]+[0] for i in range(len(indexes)-1)]
    return liste_divided


def solution_checker_ga(vrptw, solution, verbose=0):
    """
    Checks whether a solution is legitimate (i.e. meets all necessary constraints) under the context determined
    by a VRPTW instance.
    :param vrptw: VRPTW instance determining the context and rescrictions
    :param solution: Solution to be verified
    :param verbose: Level of verbosity desired
    :return: bool that indicates whether the input 'solution' is a solution or not.
    """
    penalty=0
    penalty_weight=100
    penalty_volumn=20
    penalty_time=40000
    
    nb_cust = len(vrptw.customers) # Number of customers (depot included)
    # If all customers are not visited, return False
    if set(solution) != set(range(nb_cust)):
        if verbose >= 1:
            print("All customers are not visited.")
        return -1
    # If some nodes (customers) are visited more than once (except for the depot), return False
    nb_depot = solution.count(0)
    if len(solution) != nb_depot+nb_cust-1:
        if verbose >= 1:
            print("There are customers visited more than once.")
        return -1

    if solution[0]!=0 or solution[-1]!=0:
        if verbose >= 1:
            print("Starting and ending ilegal")
        return -1

    vehicle = vrptw.vehicle
    volume, weight, cost_km = vehicle.volume, vehicle.weight, vehicle.cost_km 
    sol_routes = sol_to_list_routes(solution)
    time_matrix = vrptw.time_matrix
    customers = vrptw.customers

    for route in sol_routes:
        if verbose >= 2:
            print(f'Working on route: {route}')
        weight_cust, volume_cust = 0, 0
        for identifier in route:
            cust = customers[identifier]
            if verbose >= 3:
                print(cust)
            weight_cust += cust.request_weight
            volume_cust += cust.request_volume
            if verbose >= 2:
                print(f'weight_cust is {weight_cust} and volume_cust is {volume_cust}')
        if verbose >= 2:
            print(weight, volume, weight_cust, volume_cust)
        # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
        if weight < weight_cust or volume < volume_cust :
            if verbose >= 1:
                print(f"The weight (or volume) capacity of the vehicle ({weight}) is < to the total weight asked by customers ({identifier}) on the road ({weight_cust}):")
            penalty +=penalty_weight

        time_delivery = 0
        for index, identifier in enumerate(route[:-1]):
            if verbose >= 2:
                print(f'index={index}, id={identifier}')
            cust = customers[identifier]
            cust_plus_1 = customers[route[index+1]]
            # time_delivery += time_matrix[cust.code_customer,cust_plus_1.code_customer]
            time_delivery += time_matrix[cust.id, cust_plus_1.id]
            # If the vehicle gets there befor the beginning of the customer's time window, return False
            if time_delivery > cust_plus_1.time_window[1]:
                penalty+=penalty_time
                if verbose >= 1:
                    print(f"The vehicle is getting to late ({time_delivery}): customers' ({cust_plus_1.id}) time window's closed {cust_plus_1.time_window[1]}")
            if time_delivery < cust_plus_1.time_window[0]:
                # waiting for time window to open
                time_delivery = cust_plus_1.time_window[0]
            time_delivery += cust_plus_1.time_service
            # If the end of the delivery is after the end of the customer's time window, return False
            ##???
            if time_delivery > cust_plus_1.time_window[1]:
                if verbose >= 1:
                    print(f"The vehicle gets there after the end of the time window ({time_delivery} > {cust_plus_1.time_window[1]})")
                penalty+=penalty_time
    return penalty

#DD
def load_solomon(filename, nb_cust=None, vehicle_speed=30):
    ROOT_DIR = os.path.abspath('../')
    DATA_DIR = os.path.join(ROOT_DIR, 'data_solomon')
    DIR = os.path.join(DATA_DIR, filename)
    df = pd.read_csv("D:\Git\dir\ICO_COPY\data_solomon\simple.csv")
    
    vehicle = Vehicle(volume=sys.maxsize,
                      weight=df.at[0, 'CAPACITY'],
                      cost_km=1)
    df = df.drop('CAPACITY', axis=1)
    if nb_cust is None:
        nb_cust = len(df)
    else:
        df.drop(range(nb_cust+1, len(df)), axis=0, inplace=True)
    n = len(df)
    customers = []
    for k in range(n):
        cust = Customer(identifier=k,
                 code_customer=k,
                 latitude=df.at[k,'XCOORD'],
                 longitude=df.at[k,'YCOORD'],
                 time_window=(df.at[k,'READYTIME'], df.at[k, 'DUETIME']),
                 request_volume=0,
                 request_weight=df.at[k,'DEMAND'],
                 time_service=df.at[k,'SERVICETIME'])
        customers.append(cust)
    cust_codes = {i:i for i in range(n)}
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            x1, y1 = df.at[i, 'XCOORD'], df.at[i, 'YCOORD']
            x2, y2 = df.at[j, 'XCOORD'], df.at[j, 'YCOORD']
            distances[i,j] = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    time_matrix = (distances / vehicle_speed) * 60
    vrptw = VRPTW(customers=customers,
                  distances=distances,
                  time_matrix=time_matrix,
                  vehicle=vehicle,
                  cust_codes=cust_codes)
    return vrptw

'''
Definitions of the classes in GA
'''

#CC
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



class Modele_genetic():
    
    def __init__(self,chromosome_modele,len_chromosome,penalty_wrong_chromosome,penalty_car_road,penalty_late,penalty_volumn,penalty_weight,cost_per_car,cost_per_km):
        self.chromosome_modele=chromosome_modele
        self.len_chromosome=len_chromosome
        self.penalty_wrong_chromosome= penalty_wrong_chromosome
        self.penalty_car_road= penalty_car_road
        self.penalty_late= penalty_late
        self.penalty_volumn=penalty_volumn
        self.penalty_weight= penalty_weight
        self.cost_per_car=cost_per_car
        self.cost_per_km=cost_per_km
        self.dict_fitness={}
        
    def mutation(self,chromosome,prob):
        if random.random()<prob:
            dice=random.random()
            #SS
            if dice<0.5:
                head=random.randrange(1,len(chromosome))
                end=random.randrange(head,len(chromosome))
                tmp=chromosome[head:end]
                tmp.reverse()
                result=chromosome[:head]+tmp+chromosome[end:]

                return result
            elif dice>=0.5:
                head=random.randrange(1,len(chromosome))
                end=random.randrange(head,len(chromosome))
                tmp=chromosome[head]
                chromosome[head]=chromosome[end]
                chromosome[end]=tmp

                return chromosome
        else:
            return chromosome

    def crossover(self,parent1, parent2):

        def process_gen_repeated(copy_child1,copy_child2):
            count1=0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:#If need to fix repeated gen
                    count2=0
                    for gen2 in parent1[pos:]:#Choose next available gen
                        if gen2 not in copy_child1:
                            child1[count1] = parent1[pos:][count2]
                        count2+=1
                count1+=1

            count1=0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:#If need to fix repeated gen
                    count2=0
                    for gen2 in parent2[pos:]:#Choose next available gen
                        if gen2 not in copy_child2:
                            child2[count1] = parent2[pos:][count2]
                        count2+=1
                count1+=1

            return child1,child2

        pos=random.randrange(1,self.len_chromosome-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)

#FF  THIS PART IS RESERVED FOR DATA FROM FIL ROUGE
#     def TripDistance(self,chromosome): # Calculate the total distance of a solution indicated by a chromosome 
#         trip_distance =0
#         for i in range(len(chromosome)-1):

# #**         
#             # distance=dist_time(chromosome[i],chromosome[i+1])[0]
#             distance=vrptw.distances[i][i+1]
#             if distance>0:
#                 trip_distance+=distance 
#             else: 
#                 return -1
#         return trip_distance


#     def fitness(self,chromosome):# Calculate the fitness of a chromosome, here the fitness is determined by the reciprocal of cost
#         if tuple(chromosome) in self.dict_fitness:
#             return self.dict_fitness[tuple(chromosome)]

#         penalty_wrong_chromosome=self.penalty_wrong_chromosome
#         penalty_late=self.penalty_late
#         penalty_volumn=self.penalty_volumn
#         penalty_weight=self.penalty_weight
#         cost_per_car=self.cost_per_car
#         cost_per_km=self.cost_per_km

#         if chromosome[0]!=0 or chromosome[-1]!=0:
#             return -penalty_wrong_chromosome 

#         penalty=0
# #**
#         car='875-M-523'
#         for i in range(len(chromosome)):
#             if chromosome[i]!=0:
#                 if chromosome[i-1]==0: 
#                     cap_volumn=vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_TOTAL_VOLUME_M3'].to_list()[0]
#                     cap_weight=vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_TOTAL_WEIGHT_KG'].to_list()[0]
#                     time_now=max(vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_AVAILABLE_TIME_FROM_MIN'].to_list()[0]+dist_time(chromosome[i-1],chromosome[i])[1],customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_FROM_MIN'].to_list()[0])-dist_time(chromosome[i-1],chromosome[i])[1]
#                 else:
#                     cap_volumn-=customers[customers['CUSTOMER_CODE']==chromosome[i]]['TOTAL_VOLUME_M3'].to_list()[0]
#                     cap_weight-=customers[customers['CUSTOMER_CODE']==chromosome[i]]['TOTAL_WEIGHT_KG'].to_list()[0]
#                     time_now+=dist_time(chromosome[i],chromosome[i-1])[1]

#                     if time_now>customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_FROM_MIN'].to_list()[0] and time_now+customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'].to_list()[0]<customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_TO_MIN'].to_list()[0]:
#                         pass
#                     else:
#                         penalty+=penalty_late
#                     if cap_weight<0:
#                         penalty+=penalty_weight
#                     if cap_volumn<0:
#                         penalty+=penalty_volumn

#                     time_now+=customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'].to_list()[0]


#         cost_trip=cost_per_car*(chromosome.count(0)-1)+cost_per_km*self.TripDistance(chromosome)
#         fitness=-cost_trip-penalty
#         self.dict_fitness[tuple(chromosome)] = fitness

#         return fitness



class VRP_GA():

    def __init__(self,modele_genetic,population,rate_mutation,num_parent,num_pop,chromosome_modele,vrptw):
        self.modele_genetic = modele_genetic
        self.population = population
        self.rate_mutation = rate_mutation
        self.num_parent = num_parent
        self.num_pop=num_pop
        self.chromosome_modele = chromosome_modele
        self.vrptw = vrptw
        self.best_solution=[]


    def fitness(self,solution, omega=1000, verbose=0):
        """
        returns the total cost of the solution given for the problem given omega is the weight of each vehicle,
        1000 by default.
        """
        # data retrieval
        nb_vehicle = solution.count(0)-1
        distance_matrix = self.vrptw.distances
        cost_km = self.vrptw.vehicle.cost_km
        customers = self.vrptw.customers
        
        # solution given -> list of routes
        sol_list = sol_to_list_routes(solution)
        
        # sum of the distance of each route
        route_length = 0
        for route in sol_list:
            for i in range(len(route)-1):
                route_length += distance_matrix[route[i]][route[i+1]]
        
        # total cost calculation
        total_cost = omega*nb_vehicle + cost_km*route_length
        if verbose >= 1:
            print('Solution:', sol_list)
            print('Total cost of solution:', total_cost)

        if solution_checker_ga(self.vrptw,solution)<0:
            total_cost += self.modele_genetic.penalty_wrong_chromosome
        else:
            total_cost +=solution_checker_ga(self.vrptw,solution)
        return -total_cost

    def generate_chromosome(self,chromosome_modele):
        chromosome=[]
        for i in chromosome_modele:# chromosome_modele: a liste consiste of all of the client number only for once.
            chromosome.append(i)
        number_car=9
        for i in range(random.randint(0,number_car)):
            chromosome.append(0)
        random.shuffle(chromosome)
        chromosome.append(0)
        chromosome.insert(0,0)
        chromosome = str(chromosome).replace('0, 0, 0', '0').replace('0, 0', '0')
        chromosome = list(map(int, chromosome.strip('][').split(',')))
        return chromosome

    def initialize_population(self,num_pop,chromosome_modele):

        self.population =[self.generate_chromosome(chromosome_modele) for _ in range(num_pop)]
    
    def evolution(self): # Realize a generation, including the mating, the mutation, the elimination and the regeneration

        def binary_tournement(self):# Select certain individuals as parents by their fitness
            parents=[]            
            for i in range(self.num_parent):
                candidate=random.sample(self.population,2)

                if self.fitness(candidate[0])> self.fitness(candidate[1]):
                    if random.random()<0.95:
                        parents.append(candidate[0])
                    else:
                        parents.append(candidate[1])
                else:
                    if random.random()<0.95:
                        parents.append(candidate[1])
                    else:
                        parents.append(candidate[0])
            return parents

        def pop_crossover(self,parents):# Realize mating between parents 
            for i in range(len(parents)//2-1): 
                parent=random.sample(parents,2)
                child1,child2=self.modele_genetic.crossover(parent[0],parent[1])
                self.population.append(child1)
                self.population.append(child2)
                
            parent=random.sample(parents,2)[0]
            child1,child2=self.modele_genetic.crossover(parent,self.best_solution)
            self.population.append(child1)
            self.population.append(child2)

        
        def pop_mutation(self):# Realize mutation for all members in the population
            population_new = copy.deepcopy(self.population)
            self.population=[]
            for i in population_new:
                self.population.append(self.modele_genetic.mutation(i,self.rate_mutation))

        def polish(self):
            population_new = copy.deepcopy(self.population)
            self.population=[]
            for i in population_new:
                string = str(i).replace('0, 0, 0', '0').replace('0, 0', '0')
                i = list(map(int, string.strip('][').split(',')))
                self.population.append(i)

        def eliminate(self): # Eliminate the less strong half of the population
            list_fitness=[]
            for chromosome in self.population:
                list_fitness.append(self.fitness(chromosome))
            critere=statistics.median(list_fitness)
            best_performance=max(list_fitness)
            for i in self.population:
                if self.fitness(i)==best_performance:
                    self.best_solution=i
            while(len(self.population)>self.num_pop):
                for i in self.population:
                    if self.fitness(i)<=critere:
                        self.population.remove(i)

        def regeneration(self): # Generate new-borns to maintain the number of population remains stable
            curr_population=len(self.population)
            if self.num_pop>curr_population:
                for i in range(self.num_pop-curr_population):
                    self.population.append(self.generate_chromosome(self.chromosome_modele))
            else:
                list_fitness=[]
                for chromosome in self.population:
                    list_fitness.append(self.fitness(chromosome))
                critere=sorted(list_fitness,reverse=True)[self.num_pop-1]
                for i in self.population:
                    if self.fitness(i)<=critere:
                        self.population.remove(i)
                for i in range(self.num_pop-curr_population):
                    self.population.append(self.generate_chromosome(self.chromosome_modele))
        parents=binary_tournement(self)
        pop_crossover(self,parents)
        pop_mutation(self)
        polish(self)
        eliminate(self)
        regeneration(self)
 

'''
method to load information
'''

#FF  THIS PART IS RESERVED FOR DATA FROM FIL ROUGE
# def load_customers(customers):
#     # we supress the lines where the CUSTOMER_CODE repeat itself
#     customers = customers.drop_duplicates(subset=["CUSTOMER_CODE"], keep='first')
#     # The first customer of the list is the depot, whose id is 0.
#     id = 0
#     time_window = (depots.loc[0,"DEPOT_AVAILABLE_TIME_FROM_MIN"], depots.loc[0,"DEPOT_AVAILABLE_TIME_TO_MIN"])
#     request_volume =0
#     request_weight = 0
#     time_service = 0
#     depot = Customer(id,time_window, request_volume, request_weight, time_service)
#     list_customers = [depot]
#     # We add every new customer to the list :
#     for i, code in enumerate(customers["CUSTOMER_CODE"], start=1):
#         id = i
#         time_window = (customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_FROM_MIN"].tolist()[0], 
#                        customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_TO_MIN"].tolist()[0])
#         request_volume = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_VOLUME_M3"].tolist()[0]
#         request_weight = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_WEIGHT_KG"].tolist()[0]
#         time_service = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"].tolist()[0]
#         customer = Customer(id,time_window, request_volume, request_weight, time_service)
#         list_customers.append(customer)
#     return list_customers

# def load_vehicle(vehicles,vehicle_ids):

#     list_vehicles=[]
#     for vehicle_id in vehicle_ids:
#         volume = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_TOTAL_VOLUME_M3"].tolist()[0]
#         weight = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_TOTAL_WEIGHT_KG"].tolist()[0]
#         cost_km = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_VARIABLE_COST_KM"].tolist()[0]

#         list_vehicles.append(Vehicle(id,volume, weight, cost_km))

#     return list_vehicles



'''
load data from original excel
''' 
#FF  THIS PART IS RESERVED FOR DATA FROM FIL ROUGE
# ROOT_DIR = os.path.abspath('../')
# DATA_DIR = os.path.join(ROOT_DIR, 'data')

# CUSTOMER_DIR = os.path.join(DATA_DIR, '2_detail_table_customers.xls')
# VEHICLES_DIR = os.path.join(DATA_DIR, '3_detail_table_vehicles.xls')
# DEPOTS_DIR = os.path.join(DATA_DIR, '4_detail_table_depots.xls')
# CONSTRAINTS_DIR = os.path.join(DATA_DIR, '5_detail_table_constraints_sdvrp.xls')
# DEPOTS_DISTANCES_DIR = os.path.join(DATA_DIR, '6_detail_table_cust_depots_distances.xls')
# CUSTOMER_DISTANCES_DIR = os.path.join(DATA_DIR, '7_detail_table_cust_cust_distances.xls')


# customers = pd.read_excel(CUSTOMER_DIR)
# vehicles = pd.read_excel(VEHICLES_DIR)
# depots = pd.read_excel(DEPOTS_DIR)
# constraints = pd.read_excel(CONSTRAINTS_DIR)
# depots_dist = pd.read_excel(DEPOTS_DISTANCES_DIR)
# customers_dist = pd.read_excel(CUSTOMER_DISTANCES_DIR)

# # process customers data
# customers.drop_duplicates(['CUSTOMER_CODE'],inplace=True)
# customers.drop(['CUSTOMER_LATITUDE','CUSTOMER_LONGITUDE','NUMBER_OF_ARTICLES'],axis=1,inplace=True)
# # process vehicle data
# vehicles.drop(['ROUTE_ID','RESULT_VEHICLE_TOTAL_DRIVING_TIME_MIN','RESULT_VEHICLE_TOTAL_DELIVERY_TIME_MIN','RESULT_VEHICLE_TOTAL_ACTIVE_TIME_MIN','RESULT_VEHICLE_DRIVING_WEIGHT_KG','RESULT_VEHICLE_DRIVING_VOLUME_M3','RESULT_VEHICLE_FINAL_COST_KM'],axis=1,inplace=True)
# vehicles.drop_duplicates(['VEHICLE_CODE'],inplace=True)

# # combine the depots_dist and the customers_dist
# depots_dist.rename(columns={'DEPOT_CODE':'CUSTOMER_CODE_FROM','CUSTOMER_CODE':'CUSTOMER_CODE_TO'},inplace=True)

# depots_dist.drop(depots_dist.index[-1],inplace=True)
# depots_dist.drop(depots_dist.index[-1],inplace=True)

# for i in range(len(depots_dist)):
#     if depots_dist.at[i,'DIRECTION']=='DEPOT->CUSTOMER':
#         depots_dist.at[i,'CUSTOMER_CODE_FROM']=0
#     else:
#         depots_dist.at[i,'CUSTOMER_CODE_FROM']=depots_dist.at[i,'CUSTOMER_CODE_TO']
#         depots_dist.at[i,'CUSTOMER_CODE_TO']=0

# depots_dist.drop(['DIRECTION','CUSTOMER_NUMBER'],axis=1,inplace=True)
# all_dist=pd.concat([customers_dist,depots_dist],ignore_index=True)

# all_dist['CUSTOMER_CODE_FROM']=all_dist['CUSTOMER_CODE_FROM'].astype(int)
# all_dist['CUSTOMER_CODE_TO']=all_dist['CUSTOMER_CODE_TO'].astype(int)

# constraints.drop(constraints[constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']=='139007-1'].index,inplace=True)
# constraints.drop_duplicates(subset=['SDVRP_CONSTRAINT_CUSTOMER_CODE','SDVRP_CONSTRAINT_VEHICLE_CODE'],keep='first',inplace=True)
# constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']=constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE'].astype(int)


#**
# def dist_time(cust_from,cust_to):
#     if cust_from==0 and cust_to==0:
#         return 0,0
#     target=all_dist[all_dist['CUSTOMER_CODE_FROM']==cust_from][all_dist[all_dist['CUSTOMER_CODE_FROM']==cust_from]['CUSTOMER_CODE_TO']==cust_to]
#     if len(target)>0:
#         return target['DISTANCE_KM'].iloc[0],target['TIME_DISTANCE_MIN'].iloc[0]
#     else: 
#         return -1,-1


def init_vrpga(vrptw):
    #FF
    # chromosome_modele=customers['CUSTOMER_CODE'].unique().tolist()
    chromosome_modele=[i for i in range(1,11)]#à modifier
    len_chromosome=len(chromosome_modele)

    population=[]
    rate_mutation=0.05
    num_parent=4
    num_pop=20

    penalty_wrong_chromosome=float('inf')
    penalty_car_road=1000
    penalty_late=100
    penalty_volumn=10
    penalty_weight=10
    cost_per_car=500
    cost_per_km=10

    # vrptw=VRPTW(load_customers(customers),load_vehicle(vehicles,vehicles['VEHICLE_CODE'].unique()))

    modele_genetic=Modele_genetic(chromosome_modele,len_chromosome,penalty_wrong_chromosome,penalty_car_road,penalty_late,penalty_volumn,penalty_weight,cost_per_car,cost_per_km)
    vrp_ga=VRP_GA(modele_genetic,population,rate_mutation,num_parent,num_pop,chromosome_modele,vrptw)
    vrp_ga.initialize_population(num_pop,chromosome_modele)

    return vrp_ga
