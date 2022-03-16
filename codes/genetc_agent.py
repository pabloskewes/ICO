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
        

class modele_genetic():
    
    def __init__(self,chromosome_modele,len_chromosome):
        self.chromosome_modele=chromosome_modele
        self.len_chromosome=len_chromosome
        
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

        pos=random.randrange(1,self.len_chromosome-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)

    def distance(self,position1,position2): # Calculate the distance between two positions

        return pow((pow((map_client(position1)[0]-map_client(position2)[0]),2)+pow((map_client(position1)[1]-map_client(position2)[1]),2)),0.5)

    def TripDistance(self,chromosome): # Calculate the total distance of a solution indicated by a chromosome 

        trip_distance =0
        for i in range(len(chromosome)-1):
            trip_distance+=self.distance(chromosome[i],chromosome[i+1])
        return trip_distance

    def fitness(self,chromosome):# Calculate the fitness of a chromosome, here the fitness is determined by the reciprocal of cost

        cost_per_car=1000
        cost_route=0.2
        cost=cost_per_car*(chromosome.count(0)-1)+cost_route*self.TripDistance(chromosome)
        fitness=1/cost
        return fitness

class VRP_GA():

    def __init__(self,modele_genetic,num_generation,population,rate_mutation,num_parent,num_pop):
        self.modele_genetic = modele_genetic
        self.num_generation = num_generation
        self.population = population
        self.rate_mutation = rate_mutation
        self.num_parent = num_parent
        self.num_pop=num_pop

    def verification(chromosome):
        if chromosome :
            return True
        else:
            return False

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

    def initialize_population(self,num_pop):

        return [self.generate_chromosome() for _ in range(num_pop)]
    
    def evolution(modele_genetic,num_parent,population,rate_mutation,num_pop): # Realize a generation, including the mating, the mutation, the elimination and the regeneration

        def binary_tournement(modele_genetic,population,num_parent):# Select certain individuals as parents by their fitness
            parents=[]            
            for i in range(num_parent):
                candidate=sample(population,2)
                if modele_genetic.fitness(candidate[0])> modele_genetic.fitness(candidate[1]):
                    parents.append(candidate[0])
                else:
                    parents.append(candidate[1])
            return parents

        def pop_crossover(modele_genetic,parents,population):# Realize mating between parents 
            for i in len(parents): 
                parent=sample(parents,2)
                population.append(modele_genetic.crossover(parent[0],parent[1]))
            return population
        
        def pop_mutation(modele_genetic,population,rate_mutation):# Realize mutation for all members in the population
            for i in population:
                i=modele_genetic.mutation(i,rate_mutation)
            return population
        
        def eliminate(modele_genetic,population): # Eliminate the less strong half of the population
            list_fitness=[]
            for chromosome in population:
                list_fitness.append(modele_genetic.fitness(chromosome))
            critere=median(list_fitness)
            for i in population:
                if modele_genetic.fitness(i)<critere:
                    population.remove(i)
            return population

        def regeneration(self,population,num_pop): # Generate new-borns to maintain the number of population remains stable
            curr_population=len(population)
            for i in num_pop-curr_population:
                population.append(self.generate_chromosome())
            return population

        parents=binary_tournement(modele_genetic,population,num_parent)
        population=pop_crossover(modele_genetic,parents,population)
        population=pop_mutation(modele_genetic,population,rate_mutation)
        population=eliminate(population)
        population=regeneration(population,num_pop)




    
