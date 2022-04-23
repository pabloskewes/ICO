from mesa import Agent as MesaAgent


class BaseAgent(MesaAgent):
    def __init__(self, id, model):
        super().__init__(id, model)
        self.id = id
        self.neighborhood = model.neighborhood
        self.solution = None

    def step(self):
        pass

    def explore(self):
        N = self.neighborhood()
        self.solution = N(self.solution)
        return self.solution
