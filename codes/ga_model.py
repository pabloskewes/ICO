from pickletools import read_uint1
from pyexpat import version_info
import random
import statistics
from turtle import pen
from ga_process import*


class solution():
    def __init__(self):
        self.obj=None
        self.node_id_list = []
        self.cost_of_distance=None
        self.cost_of_time=None
        self.fitness=None
        self.route_list=[]
        self.timetable_list=[]

class Node():
    def __init__(self):
        self.id=0
        self.x_coord=0
        self.y_cooord=0
        self.demand=0
        self.depot_capacity=0
        self.start_time=0
        self.end_time=1440
        self.service_time=0


class Model():
    def __init__(self):
        self.best_solution=None
        self.demand_dict={}
        self.depot_dict={}
        self.depot_id_list=[]
        self.demand_id_list=[]
        self.sol_list=[]
        self.distance_matrix={}
        self.time_matrix={}
        self.number_of_demands=0
        self.vehicle_cap=0
        self.vehicle_speed=1
        self.pc=0.5
        self.pm=0.1
        self.popsize=100
        self.n_select=80
        self.opt_type=1

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

        
    def mutation(self,chromosome,prob):

        def inversion_mutation(chromosome_aux):#inversion globle
            chromosome=chromosome_aux
            head=random.randrange(0,len(chromosome))
            end=random.randrange(head,len(chromosome))
            tmp=chromosome[head:end]
            tmp.reverse()
            return chromosome[:head]+tmp+chromosome[end:] 

        aux=[]
        for _ in range(len(chromosome)):
            if random.random()<prob:
                aux=inversion_mutation(chromosome)
        return aux
    
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
            distance=u_find2(chromosome[i],chromosome[i+1])[0]
            if distance>0:
                trip_distance+=distance 
            else: 
                return -1
        return trip_distance

    def fitness(self,chromosome):# Calculate the fitness of a chromosome, here the fitness is determined by the reciprocal of cost
        penalty_wrong_chromosome=self.penalty_wrong_chromosome
        penalty_car_road=self.penalty_car_road
        penalty_late=self.penalty_late
        penalty_volumn=self.penalty_volumn
        penalty_weight=self.penalty_weight
        cost_per_car=self.cost_per_car
        cost_per_km=self.cost_per_km

        if chromosome[0]!=0 or chromosome[-1]!=0:
            return -penalty_wrong_chromosome 

        penalty=0
        #ROUTE ONE, CAR ONE
        vehicle_dispo=['J92-T-826','A69-O-649','875-M-523','O76-T-703','O76-T-702','A08-J-522','O78-K-074','T14-E-264','E81-M-661']
        vehicle_employe=[]
        for i in range(len(chromosome)):
            if chromosome[i]!=0 :
                if chromosome[i-1]==0:
                    vehicle_common=constraints[constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']==chromosome[1+i]]['SDVRP_CONSTRAINT_VEHICLE_CODE'].to_list()
                else:
                    for vehicle in vehicle_common:
                        if vehicle not in constraints[constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']==chromosome[i]]['SDVRP_CONSTRAINT_VEHICLE_CODE'].to_list():
                            vehicle_common.remove(vehicle)
                    # vehicle_common=[i for i in vehicle_common if i in constraints[constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']==chromosome[i]]['SDVRP_CONSTRAINT_VEHICLE_CODE'].tolist()]
                    if len(vehicle_common)<1:
                        penalty+=penalty_car_road
                        
            else:
                try:
                    vehicle_employe.append(vehicle_common[0])            
                    vehicle_dispo.remove(vehicle_common[0])
                except:
                    penalty+=penalty_car_road

        #TIME WINDOWS:penalty 
        i=1
        for car in vehicle_employe:
            time_now=max(vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_AVAILABLE_TIME_FROM_MIN'].to_list()[0]+u_find2(chromosome[i],chromosome[i-1])[1],customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_FROM_MIN'].to_list()[0])-u_find2(chromosome[i],chromosome[i-1])[1]
            while(chromosome[i]!=0):
                time_now+=u_find2(chromosome[i],chromosome[i-1])[1]
                if time_now>customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_FROM_MIN'].to_list()[0] and time_now+customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'].to_list()[0]<customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_TIME_WINDOW_TO_MIN'].to_list()[0]:
                    pass
                else:
                    penalty+=penalty_late
                time_now+=customers[customers['CUSTOMER_CODE']==chromosome[i]]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'].to_list()[0]
                i+=1
            i+=1

        #VOLUME AND WEIGHT:PENALTY
        i=1
        for car in vehicle_employe:
            cap_volumn=vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_TOTAL_VOLUME_M3'].to_list()[0]
            cap_weight=vehicles[vehicles['VEHICLE_CODE']==car]['VEHICLE_TOTAL_WEIGHT_KG'].to_list()[0]
            while(chromosome[i]!=0):
                cap_volumn-=customers[customers['CUSTOMER_CODE']==chromosome[i]]['TOTAL_VOLUME_M3'].to_list()[0]
                cap_weight-=customers[customers['CUSTOMER_CODE']==chromosome[i]]['TOTAL_WEIGHT_KG'].to_list()[0]

                if cap_volumn<0:
                    penalty+=penalty_volumn
                if cap_weight<0:
                    penalty+=penalty_weight
               
                i+=1
            i+=1

        cost_trip=cost_per_car*(chromosome.count(0)-1)+cost_per_km*self.TripDistance(chromosome)
        fitness=-cost_trip-penalty
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
            for i in range(len(parents)//2-1): 
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
                critere=sorted(list_fitness,reverse=True)[9]
                for i in self.population:
                    if self.modele_genetic.fitness(i)<critere:
                        self.population.remove(i)
            return self.population

        parents=binary_tournement(self.modele_genetic,self.population,self.num_parent)
        self.population=pop_crossover(self.modele_genetic,parents,self.population)
        self.population=pop_mutation(self.modele_genetic,self.population,self.rate_mutation)
        self.population,self.best_solution=eliminate(self.modele_genetic,self.population)
        self.population=regeneration(self)



def sol_to_list_routes(sol):
    indexes = [i for i,x in enumerate(sol) if x == 0]
    liste_divided = [sol[indexes[i]:indexes[i+1]]+[0] for i in range(len(indexes)-1)]
    return liste_divided
#[route1,route2,route3,etc], route1=[0,212,2314,4325,43524,4,1,2135,0]

def list_routes_to_sol(sol_list):
    final_sol = []
    for sol in sol_list:
        final_sol += sol[:-1]
    return final_sol + [0]
    

def solution_checker(vrptw, sol):
    nb_cust = len(vrptw.customers) # Number of customers (depot included)
    # If all customers are not visited, return False
    if set(sol)!= set(range(nb_cust)):# sol的组成：？[0,1,2,3,4,5,etc] or [0,23145,435456,13244,134565432,etc]?
        return False
    # If some nodes (customers) are visited more than once (except for the depot), return False
    nb_depot = sol.count(0)
    if len(sol) != nb_depot+nb_cust-1:
        return False
    vehicle = vrptw.vehicle
    volume, weight, cost_km = vehicle.volume, vehicle.weight, vehicle.cost_km 
    sol_routes = sol_to_list_routes(sol) #list of routes
    time_matrix = vrptw.time_matrix
    customers = vrptw.customers
    print("")

    for route in sol_routes:
        weight_cust, volume_cust, time_delivery = 0, 0, 0
        for identifier in route:
            cust = customers[identifier]
            # print(cust)
            
            weight_cust += cust.request_weight
            volume_cust += cust.request_volume
            print(weight_cust, volume_cust)
        # If the weight (or volume) capacity of the vehicle is < to the total weight asked by customers, return False
        print(weight, volume, weight_cust, volume_cust)
        if weight < weight_cust or volume < volume_cust :
            return False
        for index,identifier in enumerate(route):
            cust = [customer for customer in customers if customer.id == identifier][0]
            cust_plus_1 = [customer for customer in customers if customer.id == route[index+1]][0]
            time_delivery += time_matrix[cust.code_customer,cust_plus_1.code_customer]
            # If the vehicle gets there befor the beginning of the customer's time window, return False
            if time_delivery<cust_plus_1.time_window[0]:
                return False
            time_delivery += cust_plus_1.time_service
            # If the end of the delivery is after the end of the customer's time window, return False
            if time_delivery>cust_plus_1.time_window[1]:
                return False
    return True

def cost(vrp,sol,omega=1000):
    # data retrieval
    nb_vehicle = sol.count(0)-1
    distance_matrix = vrp.distances
    cost_km = vrp.vehicle.cost_km
    customers = vrp.customers
    
    # solution given -> list of routes
    sol_list = sol_to_list_routes(sol)
    
    # sum of the distance of each route
    route_length = 0
    for route in sol_list:
        for i in range(len(route)-1):
            route_length += distance_matrix[route[i]][route[i+1]]
    
    # total cost calculation
    total_cost = omega*nb_vehicle + cost_km*route_length
    print('Solution:', sol_list)
    print('Total cost of solution:', total_cost)
    return total_cost