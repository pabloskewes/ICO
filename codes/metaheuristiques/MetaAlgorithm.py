from abc import ABC, abstractmethod

class MetaAlgorithm(ABC):

    """
    Abstract class inherited from ABC (Abstract Base Class) class.
    It is used to create different algorithm classes.
    """
    def __init__(self):
        self.best_solution = None


    @abstractmethod
    def search(self):
        pass

    def plot_evolution(self):
        pass
