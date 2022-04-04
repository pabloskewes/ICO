import random
import statistics
import pandas as pd
import os
import sys 
from math import sqrt
import copy
import numpy as np
from base_metaheuristic import BaseMetaheuristic

from problem import Problem
'''
Category:
Identifier      Content
--------------------------------------------------------
FF              code bloques for the data from Fil rouge
SS              code bloques for the SMA
CC              code bloques of definition of classes
DD              code bloques for the simple data
'''


class GeneticAlgorithm(BaseMetaheuristic):
    def __init__(self,modele_genetic,modele_chromosome,num_parent,num_population,rate_mutation,population,fitness):
        self.best_solution = None
        self.solution = None
        self.neighborhood = None
        self.cost_list = []

        self.modele_genetic= modele_genetic
        self.population = population
        self.rate_mutation = rate_mutation
        self.num_parent= num_parent
        self.num_population= num_population
        self.modele_chromosome= modele_chromosome
        self.problem=None
        self.best_solution=[]
        self.dict_fitness={}
        self.fitness=fitness#function to be passed to a GeneticAlgorithm instance

#       parameters below should be the attributes of a problem instance
        # self.penalty_wrong_chromosome= penalty_wrong_chromosome
        # self.penalty_car_road= penalty_car_road
        # self.penalty_late= penalty_late
        # self.penalty_volumn=penalty_volumn
        # self.penalty_weight= penalty_weight
        # self.cost_per_car=cost_per_car
        # self.cost_per_km=cost_per_km


    def fit(self, problem):
            #initialize the parametres, populations
        """ Fits a metaheuristic algorithm to a specific problem """
        self.problem = problem
        # solution = problem.solution  # haven't understood this yet
        # solution.set_params(self.params['solution'])
        # self.solution = problem.solution
        # self.neighborhood = problem.neighborhood 
        return self

    # @staticmethod
    def search(self):
        """ Performs metaheuristic search """
        self.evolution()
        
    def fit_search(self, problem: Problem):
        """ Fits and search """
        return self.fit(problem).search()


    def generate_chromosome(self):
        chromosome=[]
        for i in self.modele_chromosome:
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


    def initialize_population(self):

        self.population =[self.generate_chromosome(self.modele_chromosome) for _ in range(self.num_population)]


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
                child1,child2=self.crossover(parent[0],parent[1])
                self.population.append(child1)
                self.population.append(child2)
                
            parent=random.sample(parents,2)[0]
            child1,child2=self.crossover(parent,self.best_solution)
            self.population.append(child1)
            self.population.append(child2)
        
        def pop_mutation(self):# Realize mutation for all members in the population
            population_new = copy.deepcopy(self.population)
            self.population=[]
            for i in population_new:
                self.population.append(self.mutation(i,self.rate_mutation))

        def optimize(self):
            population_opt = copy.deepcopy(self.population)
            self.population=[]
            for i in population_opt:
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
                    self.population.append(self.generate_chromosome(self.modele_chromosom))
            else:
                list_fitness=[]
                for chromosome in self.population:
                    list_fitness.append(self.fitness(chromosome))
                critere=sorted(list_fitness,reverse=True)[self.num_pop-1]
                for i in self.population:
                    if self.fitness(i)<=critere:
                        self.population.remove(i)
                for i in range(self.num_pop-curr_population):
                    self.population.append(self.generate_chromosome(self.modele_chromosom))
        parents=binary_tournement(self)
        pop_crossover(self,parents)
        pop_mutation(self)
        optimize(self)
        eliminate(self)
        regeneration(self)




        





    def plot_evolution(self):
        raise NotImplementedError










def sol_to_list_routes(sol):
    """
    Transforms [0, x1, x2, 0, x3, 0, x4, x5, x6, 0] into [[0, x1, x2, 0], [0, x3, 0], [0, x4, x5, x6, 0]].
    """
    indexes = [i for i, x in enumerate(sol) if x == 0]
    liste_divided = [sol[indexes[i]:indexes[i+1]]+[0] for i in range(len(indexes)-1)]
    return liste_divided

# solution checker belongs to the proble classes
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
    
    def __init__(self,modele_chromosom,penalty_wrong_chromosome,penalty_late,penalty_volumn,penalty_weight,cost_per_car,cost_per_km):
        self.modele_chromosom=modele_chromosom
        # parametres below are defined by a certain problem so they should be passed from a problem instance. 
        self.penalty_wrong_chromosome= penalty_wrong_chromosome
        # self.penalty_car_road= penalty_car_road
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

        pos=random.randrange(1,len(self.modele_chromosom)-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)



class VRP_GA():

    def __init__(self,modele_genetic,population,rate_mutation,num_parent,num_pop,modele_chromosom,vrptw):
        self.modele_genetic = modele_genetic
        self.population = population
        self.rate_mutation = rate_mutation
        self.num_parent = num_parent
        self.num_pop=num_pop
        self.modele_chromosom = modele_chromosom
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


    
'''
method to load information
'''

def init_vrpga(vrptw):
    #FF
    # modele_chromosom=customers['CUSTOMER_CODE'].unique().tolist()
    modele_chromosom=[i for i in range(1,11)]#à modifier
    population=[]
    rate_mutation=0.05
    num_parent=4
    num_pop=20

    penalty_wrong_chromosome=float('inf')
    penalty_car_road=1000
    penalty_late=40000
    penalty_volumn=20
    penalty_weight=100
    cost_per_car=1000
    cost_per_km=1

    # vrptw=VRPTW(load_customers(customers),load_vehicle(vehicles,vehicles['VEHICLE_CODE'].unique()))

    modele_genetic=Modele_genetic(modele_chromosom,penalty_wrong_chromosome,penalty_late,penalty_volumn,penalty_weight,cost_per_car,cost_per_km)
    vrp_ga=VRP_GA(modele_genetic,population,rate_mutation,num_parent,num_pop,modele_chromosom,vrptw)
    vrp_ga.initialize_population(num_pop,modele_chromosom)

    return vrp_ga
