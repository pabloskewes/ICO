from mesa import Agent as MesaAgent


class BaseAgent(MesaAgent):
    def __init__(self, id, model):
        super().__init__(id, model)
        self.id = id

    def step(self):
        pass
