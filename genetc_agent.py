import random
from random import randrange

class Problem_Genetic(Object):
    
    def __init__(self,genes,individuals_length,decode,fitness):
        self.genes=genes
        self.individuals_length=individuals_length
        self.decode=decode
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

def fitnessVRP(chromosome):
# Contraintes pour les chromosomes:
# Il faut qu'un chromosome réaliser tous les divrasons. Sinon il aura une fitness négatiff
# 
#
#
    for i in customer:
        if i not in chromosome:
            return -1

    def TripDistance(chromosome):
        trip_distance =0
        for i in range(len(chromosome)-1):
            trip_distance+=distance(chromosome[i],chromosome[i+1])
        return trip_distance

    def distance(position1,position2):
        return pow((pow((map_client(position1)[0]-map_client(position2)[0]),2)+pow((map_client(position1)[1]-map_client(position2)[1]),2)),0.5)

    fitness=cost_per_car*(chromosome.count(0)-1)+TripDistance(chromosome)
    return fitness

cost_per_car=100
customer=[]
map_client=[]


def genetic_algorithm_t(Problem_Genetic,k,opt,ngen,size,rate_cross,rate_mutate):

    def initialize_population(Problem_Genetic,size):
        def generate_chromosome():
            chromosome=[]
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            return chromosome
        return [generate_chromosome() for _ in range(size)]
        