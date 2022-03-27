from mesa import Agent, Model
from tabu import tabu_method

class TabuAgent(Agent):
    def __init__(self, id_agent, sma):
        super().__init__(id_agent, sma)
        self.name = "Agent "+str(id_agent)        
        
    def __str__(self):
        return "Tabu agent : " + self.name
    
    def step(self, vrptw, sol):
        tabu_method(vrptw, sol)

# proposition de modèle
class SMA(Model):
    def __init__(self, vrptw):
        self.schedule = []
        self.vrptw = vrptw
        
    def step(self, sol):
        self.schedule.step(self.vrptw, sol)
