from ast import In
from email.mime import application
from logging import critical
import random
from random import randrange, sample
from statistics import median



class client():
    def __init__(self,id,weight,volumn,time_window,vehicles,possition,service_time):
        self.id=id
        self.weight=weight
        self.volumn=volumn
        self.time_window=time_window
        self.vehicles=vehicles
        self.possition=possition
        self.service_time=service_time

class vehicle():
    def __init__(self,id,cap_weight,cap_volumn) :
        self.id=id
        self.cap_weight=cap_weight
        self.cap_volumn=cap_volumn
        

class Problem_Genetic():
    
    def __init__(self,genes,individuals_length,fitness):
        self.genes=genes
        self.individuals_length=individuals_length
        self.fitness=fitness
        
    def mutation(self,chromosome,prob):

        def inversion_mutation(chromosome_aux):#inversion globle
            chromosome=chromosome_aux
            head=randrange(0,len(chromosome))
            end=randrange(head,len(chromosome))
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

            return [child1,child2]

        pos=random.randrange(1,self.individuals_length-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)


#   Contraintes-faut-il-repecter pour les chromosomes:
#   Il faut qu'un chromosome réaliser tous les divrasons. Sinon il aura une fitness négatiff
#   1 - respect du time window
#   2 - capacité camion kg
#   3 - capacité camion m3
#   4 - chaque customer doit etre visité une fois
#   5 - 1 route = 1 camion


# tri la population par ordre croissant de la fitness
# sumpprimer la moins forte chromosome

def verification(chromosome):
    if chromosome :
        return True
    else:
        return False

def genetic_algorithm(Problem_Genetic,target_optimisation,num_generation,population,rate_cross,rate_mutation,num_parent):

    def initialize_population(Problem_Genetic,size):
        def generate_chromosome():
            chromosome=[]
            for i in chromosome_modele:# chromosome_modele: a liste consiste of all of the client number only for once.
                chromosome.append(i)
            number_car=11
            for i in range(random.randint(0,number_car-1)):
                chromosome.append(0)
            random.shuffle(chromosome)
            chromosome.append(0)
            chromosome.insert(0,0)

            return chromosome
        return [generate_chromosome() for _ in range(size)]
    
    def evolution(Problem_Genetic,num_parent,population,rate_mutation):
    #   selection      
    #   crossover
    #   mutation
    #   elimination
        def binary_tournement(Problem_Genetic,population,num_parent):
            parents=[]            
            for i in range(num_parent):
                candidate=random.sample(population,2)
                if fitness(candidate[0])> fitness(candidate[1]):
                    parents.append(candidate[0])
                else:
                    parents.append(candidate[1])
            return parents

        def pop_crossover(Problem_Genetic,parents,population):
            for i in len(parents):#生四个
                parent=random.sample(parents,2)
                population.append(Problem_Genetic.crossover(parent[0],parent[1]))
            return population
        
        def pop_mutation(Problem_Genetic,population,rate_mutation):
            for i in population:
                i=Problem_Genetic.mutation(i,rate_mutation)
            return population
        
        def eliminate(population): #num_elimination=num_parent: num_pop stable
            list_fitness=[]
            for chromosome in population:
                list_fitness.append(fitness(chromosome))
            critere=median(list_fitness)
            for i in population:
                if fitness(i)<critere:
                    population.remove(i)
        
        def regeneration(population):
            
                


def distance(client1,client2):
    return pow((pow((map_client(position1)[0]-map_client(position2)[0]),2)+pow((map_client(position1)[1]-map_client(position2)[1]),2)),0.5)

def TripDistance(chromosome):

    trip_distance =0
    for i in range(len(chromosome)-1):
        trip_distance+=distance(chromosome[i],chromosome[i+1])
    return trip_distance

def fitness(chromosome):

    fitness=cost_per_car*(chromosome.count(0)-1)+TripDistance(chromosome)
    return fitness


    

for g in num_generation:

    #   fitness evaluation
    #   demande met verification
