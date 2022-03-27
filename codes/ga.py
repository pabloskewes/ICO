from statistics import median
from ga_model import*
from ga_process import*
from ga_load import*

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

    for i in range(num_generation):
        vrp_ga.evolution()

    return vrp_ga

