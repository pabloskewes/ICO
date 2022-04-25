from metaheuristics.tabu_search import TabuSearch
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

from loading_models import load_solomon
from vrptw import VRPTW


class TabuAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        self.tabu_model = TabuSearch()
        self.tabu_model.fit(model.problem)
        
        self.name = "TabuSearch Agent"
        self.solution = None
        
    def step(self):
        self.solution = self.tabu_model.search()


class TestModel(Model):
    def __init__(self, problem, num_agents=1, width=5, height=5):
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.problem = problem
        
        # Create agents
        for i in range(self.num_agents):
            a = TabuAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            
    def step(self):
        self.schedule.step()
        
    def print_solution(self):
        for a in self.schedule.agents:
            print(a.name, ":", a.solution, "cost :", a.solution.cost())



# Example

context = load_solomon('simple.csv', nb_cust=10, vehicle_speed=100)
vrptw = VRPTW(context)
model = TestModel(vrptw, num_agents)
model.step()
model.print_solution()