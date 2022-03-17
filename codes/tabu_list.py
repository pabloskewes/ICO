from solution import generate_cost_function
from neighborhood import Neighborhood


class TabuList:
    def __init__(self):
        self.tabu_list = []
        self.size = 0

    def push(self, e):
        self.tabu_list.append(e)
        self.size += 1

    def remove_first(self):
        self.tabu_list = self.tabu_list[1:]
        self.size -= 1

    def contains(self, e):
        return e in self.tabu_list

# TODO: ADD VERBOSITY LEVELS FOR DEBUGGING
def tabu_method(vrptw, sol, lower_bound=100, max_iter=100, max_tabu=10, verbose=0):
    # initial setup
    best_sol = sol
    best_candidate = sol
    n_iter = 0
    T = TabuList()
    T.push(sol)
    best_iter = 0
    cost = generate_cost_function(vrptw, omega=1000, verbose=0)
    N = Neighborhood(vrptw)

    def aspiration(sol, neighborhood, A):
        best_sol = sol
        for neighbor in neighborhood:
            if cost(neighbor) <= A:
                best_sol = neighbor
        return best_sol

        # looking for the best solution

    # if the best solution is worse than the lower bound and
    # we have not reached the max iterations, we continue
    while (cost(best_sol) > lower_bound) and ((n_iter - best_iter) < max_iter):

        n_iter += 1
        # TODO: No sé qué solución se supone que va acá pero tú vay a cachar Nico
        neighborhood = [N.shuffle(best_candidate) for _ in range(0, 9)]
        # << generated a neighborhood using every neighborhood function

        # we look for the best candidate in the neighborhood
        best_candidate = neighborhood[0]
        for neighbor in neighborhood:
            if (not T.contains(neighbor)) and (cost(neighbor) < cost(best_candidate)):
                best_candidate = neighbor

        # if the best candidate is in the tabu list, we call the aspiration function
        if T.contains(best_candidate):
            best_candidate = aspiration(best_candidate, neighborhood, cost(best_sol) - 1)

        # once we found it, we compare it with our actual best solution
        if cost(best_candidate) < cost(best_sol):
            best_sol = best_candidate
            best_iter = n_iter

        # finally we add the candidate to the tabu list
        T.push(best_candidate)

        # if we reached the maximum of tabu elements, we erase the first added
        if T.size > max_tabu:
            T.remove_first()

    return best_sol
