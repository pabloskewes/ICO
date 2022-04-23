from ..tabu_search import TabuList


class TabuAgent:
    def __init__(self, max_tabu: int = 100, max_iter: int = 100, tabu_mode: str = 'default'):
        self.max_tabu = max_tabu
        self.tabu_mode = tabu_mode
        self.T = TabuList(mode=self.tabu_mode)
        self.max_iter = max_iter
        self.last_solution = None

    def explore(self, neighborhood, init_sol=None):
        if init_sol is None:
            init_sol = neighborhood.initial_solution()

        if self.last_solution is None:
            self.last_solution = init_sol

        self.T.push(init_sol)
        new_solution = neighborhood(init_sol)
        n_cycle = 0
        while self.T.contains(new_solution):
            new_solution = neighborhood(init_sol)
            if n_cycle == self.max_iter:
                new_solution = self.last_solution
                break
            n_cycle += 1
        if new_solution.cost() < self.last_solution.cost():
            self.last_solution = new_solution

        return new_solution
