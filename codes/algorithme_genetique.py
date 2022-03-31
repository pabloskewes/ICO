from statistics import median
from pickletools import read_uint1
from pyexpat import version_info
import random
import statistics
from turtle import pen
import pandas as pd
import os

'''
Definitions of the classes in GA
'''

class Vehicle:
    def __init__(self,id,volume, weight, cost_km):
        self.id = id
        self.volume = volume
        self.weight = weight
        self.cost_km = cost_km

    def __str__(self):
        return f'Vehicle of volume {self.volume}, weight {self.weight}'

    
class Customer:
    def __init__(self, code,time_window, request_volume, request_weight, time_service):
        self.id = code
        self.time_window = time_window
        self.request_volume = request_volume
        self.request_weight = request_weight
        self.time_service = time_service
        
    def __str__(self):
        return f'This customer\'s id is {self.id}, its code_customer is {self.code_customer}, ' \
               f'its latitude is {self.latitude}, its longitude is {self.longitude}, ' \
               f'its time window is {self.time_window}, its volume requested is {self.request_volume},' \
               f'its weight requested is {self.request_weight}, its time service is {self.time_service}.'

class VRP:
    """
    A class that store all the data
    """
    def __init__(self, customers,vehicle):
        self.customers = customers
        # self.distances = distances
        # self.time_matrix = time_matrix
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
            head=random.randrange(0,len(chromosome))
            end=random.randrange(head,len(chromosome))
            tmp=chromosome[head:end]
            tmp.reverse()

            result=chromosome[:head]+tmp+chromosome[end:]
            if result!= chromosome:
                print('really helped')
            return result
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

    def TripDistance(self,chromosome): # Calculate the total distance of a solution indicated by a chromosome 
        
        trip_distance =0
        for i in range(len(chromosome)-1):
            distance=dist_time(chromosome[i],chromosome[i+1])[0]
            if distance>0:
                trip_distance+=distance 
            else: 
                return -1
        return trip_distance

    def fitness(self,chromosome):# Calculate the fitness of a chromosome, here the fitness is determined by the reciprocal of cost
        if tuple(chromosome) in self.dict_fitness:
            return self.dict_fitness[tuple(chromosome)]

        penalty_wrong_chromosome=self.penalty_wrong_chromosome
        penalty_late=self.penalty_late
        penalty_volumn=self.penalty_volumn
        penalty_weight=self.penalty_weight
        cost_per_car=self.cost_per_car
        cost_per_km=self.cost_per_km

        if chromosome[0]!=0 or chromosome[-1]!=0:
            return -penalty_wrong_chromosome 

        penalty=0

        car='875-M-523'
        for i in range(len(chromosome)):
            if chromosome[i]!=0:
                if chromosome[i-1]==0: 
                    cap_volumn=vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_TOTAL_VOLUME_M3'].to_list()[0]
                    cap_weight=vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_TOTAL_WEIGHT_KG'].to_list()[0]
                    time_now=max(vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_AVAILABLE_TIME_FROM_MIN'].to_list()[0]+dist_time(chromosome[i-1],chromosome[i])[1],customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_FROM_MIN'].to_list()[0])-dist_time(chromosome[i-1],chromosome[i])[1]
                else:
                    cap_volumn-=customers[customers['CUSTOMER_CODE']==chromosome[i]]['TOTAL_VOLUME_M3'].to_list()[0]
                    cap_weight-=customers[customers['CUSTOMER_CODE']==chromosome[i]]['TOTAL_WEIGHT_KG'].to_list()[0]
                    time_now+=dist_time(chromosome[i],chromosome[i-1])[1]

                    if time_now>customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_FROM_MIN'].to_list()[0] and time_now+customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'].to_list()[0]<customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_TO_MIN'].to_list()[0]:
                        pass
                    else:
                        penalty+=penalty_late
                    if cap_weight<0:
                        penalty+=penalty_weight
                    if cap_volumn<0:
                        penalty+=penalty_volumn

                    time_now+=customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'].to_list()[0]


        cost_trip=cost_per_car*(chromosome.count(0)-1)+cost_per_km*self.TripDistance(chromosome)
        fitness=-cost_trip-penalty
        self.dict_fitness[tuple(chromosome)] = fitness

        return fitness

class VRP_GA():

    def __init__(self,modele_genetic,num_generation,population,rate_mutation,num_parent,num_pop,chromosome_modele,vrp):
        self.modele_genetic = modele_genetic
        self.num_generation = num_generation
        self.population = population
        self.rate_mutation = rate_mutation
        self.num_parent = num_parent
        self.num_pop=num_pop
        self.chromosome_modele = chromosome_modele
        self.vrp = vrp
        self.best_solution=[]

    def verification(self,chromosome):
        if chromosome :
            return True
        else:
            return False

    def generate_chromosome(self,chromosome_modele):
        chromosome=[]
        
        for i in chromosome_modele:# chromosome_modele: a liste consiste of all of the client number only for once.
            chromosome.append(i)
        number_car=9
        for i in range(random.randint(0,number_car-1)):
            chromosome.append(0)
        random.shuffle(chromosome)
        chromosome.append(0)
        chromosome.insert(0,0)

        return chromosome

    def initialize_population(self,num_pop,chromosome_modele):

        self.population =[self.generate_chromosome(chromosome_modele) for _ in range(num_pop)]
    
    def evolution(self): # Realize a generation, including the mating, the mutation, the elimination and the regeneration

        def binary_tournement(modele_genetic,population,num_parent):# Select certain individuals as parents by their fitness
            parents=[]            
            for i in range(num_parent):
                candidate=random.sample(population,2)

                if modele_genetic.fitness(candidate[0])> modele_genetic.fitness(candidate[1]):
                    parents.append(candidate[0])
                else:
                    parents.append(candidate[1])
            return parents

        def pop_crossover(modele_genetic,parents,population):# Realize mating between parents 
            for i in range(len(parents)//2): 
                parent=random.sample(parents,2)
                population.append(modele_genetic.crossover(parent[0],parent[1])[0])
                population.append(modele_genetic.crossover(parent[0],parent[1])[1])
            return population
        
        def pop_mutation(modele_genetic,population,rate_mutation):# Realize mutation for all members in the population
            for i in population:
                i=modele_genetic.mutation(i,rate_mutation)
            return population
        
        def eliminate(modele_genetic,population): # Eliminate the less strong half of the population
            list_fitness=[]
            for chromosome in population:
                list_fitness.append(modele_genetic.fitness(chromosome))
            critere=statistics.median(list_fitness)
            best_performance=max(list_fitness)
            for i in population:
                if modele_genetic.fitness(i)==best_performance:
                    best_solution=i
            for i in population:
                if modele_genetic.fitness(i)<critere:
                    population.remove(i)
            return population, best_solution

        def regeneration(self): # Generate new-borns to maintain the number of population remains stable
            curr_population=len(self.population)
            if self.num_pop>curr_population:
                for i in range(self.num_pop-curr_population):
                    self.population.append(self.generate_chromosome(self.chromosome_modele))
            else:
                list_fitness=[]
                for chromosome in self.population:
                    list_fitness.append(self.modele_genetic.fitness(chromosome))
                critere=sorted(list_fitness,reverse=True)[self.num_pop-1]
                for i in self.population:
                    if self.modele_genetic.fitness(i)<=critere:
                        self.population.remove(i)
                for i in range(self.num_pop-curr_population):
                    self.population.append(self.generate_chromosome(self.chromosome_modele))

            return self.population
        print("binary_tournement,num_pop:",len(self.population))
        parents=binary_tournement(self.modele_genetic,self.population,self.num_parent)
        print("pop_crossover,num_pop:",len(self.population))
        self.population=pop_crossover(self.modele_genetic,parents,self.population)
        print("pop_mutation,num_pop:",len(self.population))
        self.population=pop_mutation(self.modele_genetic,self.population,self.rate_mutation)
        print("elimination,num_pop:",len(self.population))
        self.population,self.best_solution=eliminate(self.modele_genetic,self.population)
        print("regeneration,num_pop:",len(self.population))
        self.population=regeneration(self)
        print("End of evolution, fitness best solution:",self.modele_genetic.fitness(self.best_solution),'\n')



'''
method to load information
'''


def load_customers(customers):
    # we supress the lines where the CUSTOMER_CODE repeat itself
    customers = customers.drop_duplicates(subset=["CUSTOMER_CODE"], keep='first')
    # The first customer of the list is the depot, whose id is 0.
    id = 0
    time_window = (depots.loc[0,"DEPOT_AVAILABLE_TIME_FROM_MIN"], depots.loc[0,"DEPOT_AVAILABLE_TIME_TO_MIN"])
    request_volume =0
    request_weight = 0
    time_service = 0
    depot = Customer(id,time_window, request_volume, request_weight, time_service)
    list_customers = [depot]
    # We add every new customer to the list :
    for i, code in enumerate(customers["CUSTOMER_CODE"], start=1):
        id = i
        time_window = (customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_FROM_MIN"].tolist()[0], 
                       customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_TIME_WINDOW_TO_MIN"].tolist()[0])
        request_volume = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_VOLUME_M3"].tolist()[0]
        request_weight = customers[customers["CUSTOMER_CODE"]==code]["TOTAL_WEIGHT_KG"].tolist()[0]
        time_service = customers[customers["CUSTOMER_CODE"]==code]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"].tolist()[0]
        customer = Customer(id,time_window, request_volume, request_weight, time_service)
        list_customers.append(customer)
    return list_customers

def load_vehicle(vehicles,vehicle_ids):

    list_vehicles=[]
    for vehicle_id in vehicle_ids:
        volume = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_TOTAL_VOLUME_M3"].tolist()[0]
        weight = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_TOTAL_WEIGHT_KG"].tolist()[0]
        cost_km = vehicles[vehicles["VEHICLE_CODE"]==vehicle_id]["VEHICLE_VARIABLE_COST_KM"].tolist()[0]

        list_vehicles.append(Vehicle(id,volume, weight, cost_km))

    return list_vehicles


'''
load data from excel
''' 
# def ga_process():
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CUSTOMER_DIR = os.path.join(DATA_DIR, '2_detail_table_customers.xls')
VEHICLES_DIR = os.path.join(DATA_DIR, '3_detail_table_vehicles.xls')
DEPOTS_DIR = os.path.join(DATA_DIR, '4_detail_table_depots.xls')
CONSTRAINTS_DIR = os.path.join(DATA_DIR, '5_detail_table_constraints_sdvrp.xls')
DEPOTS_DISTANCES_DIR = os.path.join(DATA_DIR, '6_detail_table_cust_depots_distances.xls')
CUSTOMER_DISTANCES_DIR = os.path.join(DATA_DIR, '7_detail_table_cust_cust_distances.xls')


customers = pd.read_excel(CUSTOMER_DIR)
vehicles = pd.read_excel(VEHICLES_DIR)
depots = pd.read_excel(DEPOTS_DIR)
constraints = pd.read_excel(CONSTRAINTS_DIR)
depots_dist = pd.read_excel(DEPOTS_DISTANCES_DIR)
customers_dist = pd.read_excel(CUSTOMER_DISTANCES_DIR)

# process customers data
customers.drop_duplicates(['CUSTOMER_CODE'],inplace=True)
customers.drop(['CUSTOMER_LATITUDE','CUSTOMER_LONGITUDE','NUMBER_OF_ARTICLES'],axis=1,inplace=True)
# process vehicle data
vehicles.drop(['ROUTE_ID','RESULT_VEHICLE_TOTAL_DRIVING_TIME_MIN','RESULT_VEHICLE_TOTAL_DELIVERY_TIME_MIN','RESULT_VEHICLE_TOTAL_ACTIVE_TIME_MIN','RESULT_VEHICLE_DRIVING_WEIGHT_KG','RESULT_VEHICLE_DRIVING_VOLUME_M3','RESULT_VEHICLE_FINAL_COST_KM'],axis=1,inplace=True)
vehicles.drop_duplicates(['VEHICLE_CODE'],inplace=True)

# combine the depots_dist and the customers_dist
depots_dist.rename(columns={'DEPOT_CODE':'CUSTOMER_CODE_FROM','CUSTOMER_CODE':'CUSTOMER_CODE_TO'},inplace=True)

depots_dist.drop(depots_dist.index[-1],inplace=True)
depots_dist.drop(depots_dist.index[-1],inplace=True)

for i in range(len(depots_dist)):
    if depots_dist.at[i,'DIRECTION']=='DEPOT->CUSTOMER':
        depots_dist.at[i,'CUSTOMER_CODE_FROM']=0
    else:
        depots_dist.at[i,'CUSTOMER_CODE_FROM']=depots_dist.at[i,'CUSTOMER_CODE_TO']
        depots_dist.at[i,'CUSTOMER_CODE_TO']=0

depots_dist.drop(['DIRECTION','CUSTOMER_NUMBER'],axis=1,inplace=True)
all_dist=pd.concat([customers_dist,depots_dist],ignore_index=True)

all_dist['CUSTOMER_CODE_FROM']=all_dist['CUSTOMER_CODE_FROM'].astype(int)
all_dist['CUSTOMER_CODE_TO']=all_dist['CUSTOMER_CODE_TO'].astype(int)

# process the constraints data
constraints.drop(constraints[constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']=='139007-1'].index,inplace=True)
constraints.drop_duplicates(subset=['SDVRP_CONSTRAINT_CUSTOMER_CODE','SDVRP_CONSTRAINT_VEHICLE_CODE'],keep='first',inplace=True)
constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']=constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE'].astype(int)

    # return customers,vehicles,depots,constraints,all_dist


def dist_time(cust_from,cust_to):
    if cust_from==0 and cust_to==0:
        return 0,0
    target=all_dist[all_dist['CUSTOMER_CODE_FROM']==cust_from][all_dist[all_dist['CUSTOMER_CODE_FROM']==cust_from]['CUSTOMER_CODE_TO']==cust_to]
    if len(target)>0:
        return target['DISTANCE_KM'].iloc[0],target['TIME_DISTANCE_MIN'].iloc[0]
    else: 
        return -1,-1

'''
'''

def init_vrpga():
    chromosome_modele=customers['CUSTOMER_CODE'].unique().tolist()
    len_chromosome=len(chromosome_modele)

    num_generation=2
    population=[]
    rate_mutation=0.05
    num_parent=4
    num_pop=20

    penalty_wrong_chromosome=1000000
    penalty_car_road=1000
    penalty_late=100
    penalty_volumn=10
    penalty_weight=10
    cost_per_car=500
    cost_per_km=10

    vrp=VRP(load_customers(customers),load_vehicle(vehicles,vehicles['VEHICLE_CODE'].unique()))
    modele_genetic=Modele_genetic(chromosome_modele,len_chromosome,penalty_wrong_chromosome,penalty_car_road,penalty_late,penalty_volumn,penalty_weight,cost_per_car,cost_per_km)
    vrp_ga=VRP_GA(modele_genetic,num_generation,population,rate_mutation,num_parent,num_pop,chromosome_modele,vrp)
    vrp_ga.initialize_population(num_pop,chromosome_modele)

    return vrp_ga

