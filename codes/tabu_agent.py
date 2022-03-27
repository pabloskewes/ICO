from mesa import Agent, Model
from tabou import tabu_method
from recuit_simule import simulated_annealing
from ga import *

# Agent ordonnanceur Tabou
class TabuAgent(Agent):
    def __init__(self, id_agent, sma):
        super().__init__(id_agent, sma)
        self.name = "Agent "+str(id_agent)        
        
    def __str__(self):
        return "Tabu agent : " + self.name
    
    def step(self, vrptw, sol):
        tabu_method(vrptw, sol)


# Agent ordonnanceur Recuit Simulé
class AgentRS(Agent):
    #le constructeur
    def __init__(self,id,model):
        super().__init__(id,model)
        self.name = "AgentRS"+str(self.id)

    def __str__(self):
        return "RS agent : " + self.name

    def step(self, vrptw, sol, T0):
        simulated_annealing(vrptw, sol, T0, cooling_factor=0.9, max_cycle_iter=100)

class Genetic(Agent):

    def __init__(self,id,model):
        super().__init__(id,model)
        self.name = "Agent"+str(id)
        self.vrp_ga=init_vrpga()
        self.solution=self.vrp_ga.best_solution

    def __str__(self):
        return "GA agent : "+self.name

    def step(self):
        self.vrp_ga.evolution()

# proposition de modèle
class SMA(Model):
    def __init__(self, vrptw):
        self.schedule = []
        self.vrptw = vrptw
        
    def step(self, sol):
        self.schedule.step(self.vrptw, sol)
