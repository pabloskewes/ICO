from ast import In
from email.mime import application
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
#   Contraintes-faut-il-repecter pour les chromosomes:
#   Il faut qu'un chromosome réaliser tous les divrasons. Sinon il aura une fitness négatiff
#   1 - respect du time window
#   2 - capacité camion kg
#   3 - capacité camion m3
#   4 - chaque customer doit etre visité une fois
#   5 - 1 route = 1 camion

#   data structure correspondant:
#   chromosome: [0,5,12231,31,45567,1,3456,67689,0], len(chromosome)≈num_client=573 
#   question sur ce probleme: Parce que le nombre de camion n'est pas fixé entre les chromosomes, est-ce que le crossover est utile? Si le modele ne converge pas ou il converge très doucement?

#对计算能力的质疑...
#1. time window:
#   计算当前时间，如果超出服务时间，则该基因死亡（或者削弱？）是否进行下一家的配送？完整配送所有客户是不是必须的？
#   如果完整配送是必须的：则基因死亡
#   如果完整配送不是必须的，则基因收到惩罚：fitness收到下降效果

#3. Un route, un camion:
#   针对该contrainte的两种思路：排除法vs生成法
#   排除法：先生成chromo,在fitness计算部分 kill掉不符合contrainte的chromos，
#   优点：生成代码简洁经典。缺点：时间成本高，kill()函数还没想好怎么写:(soit 使用交集，时间复杂度取决于每个客户链接的route)
#   生成法：在生成chromos阶段就使用该contraint生成
#   优点：在fitness阶段无需针对此contrainte对chromosome处理
#   缺点：生成chromosome的方法出现限制，与GA的基本观念有出入，而且目前还没有怎么生成的想法
#   结论：综合分析结果，采用经典排除法 
#   
    def intersection(camion_list1,camion_list2):

        intersection_list=[]
        if len(camion_list1)<len(camion_list2):
            for i in camion_list1:
                if i in camion_list2:
                    intersection_list.append(i)
        else:
            for i in camion_list2:
                if i in camion_list1:
                    intersection_list.append(i)   

        return intersection_list    

#   prise en main de contrainte 4: chaque customer doit etre visité une fois
    def contrainte_4(chromosome):

        client_delivré=set(chromosome)
        if not len(client_delivré)==574: # len(df2['CUSTOMER_CODE'].unique())=573, on prise en compte de le point de départ
            return -1
    
        if not len(chromosome)-chromosome.count(0)==573:
            return -1
        
#   prise en main des contraintes 1,2,3,5:
    def contraintes_1235(chromosome):
        
        for i in range(len(chromosome)):
            # contrainte 5
            # on a besion d'un chromosome qui satisfait tout les contrainte en même temps. Alors concernat la contrainte de capacité, il faut fixer le camion utilisé dans une route d'abord
            # problem reste: si il ya plusieur camion dans la liste camion_partagé à la fin d"une route?
            if chromosome[i] == 0:
                if not i ==len(chromosome)-1:
                    camion_partagé=chromosome[i+1].camion_list #chromosome_test[i+1].camion_list ça veut dire la liste de camion qui peut passer le client chromosome[i]
                else:
                    continue
            else:
                camion_partagé=intersection(camion_partagé,chromosome[i].camion_list)
            if len(camion_partagé)==0:
                return -1

        if len(camion_partagé)==1: #cas idéal
            
        else:
            return 0

            # contrainte 1
            
#2. contrainte 2,3
#   for i in range(head,end):
#       capacité.kg-=client[i].weight
#       capacité.m3-=client[i].volumn
#       if capacité.kg<0 or capacité.m3:
#           kill chromosome


    contrainte_4(chromosome)
    contraintes_1235(chromosome)

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
        