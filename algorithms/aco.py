import numpy as np

class Ant:
    def __init__(self, graph, alpha, beta):
        self.graph = graph
        self.current_node = np.random.randint(graph.num_nodes)
        self.path = [self.current_node]
        self.total_distance = 0
        self.unvisited_nodes = set(range(self.graph.num_nodes))
        self.unvisited_nodes.remove(self.current_node)
        self.alpha = alpha
        self.beta = beta

    def select_next_node(self):
        probabilities = np.zeros(self.graph.num_nodes)

        for node in self.unvisited_nodes:
            distance = self.graph.get_distance(self.current_node, node)
            pheromone = self.graph.get_pheromone(self.current_node, node)

            heuristic = 1 / distance
            probabilities[node] = (pheromone ** self.alpha) * (heuristic ** self.beta)

        probabilities /= probabilities.sum()
        next_node = np.random.choice(range(self.graph.num_nodes), p=probabilities)
        return next_node

    def move(self):
        next_node = self.select_next_node()
        self.path.append(next_node)
        self.total_distance += self.graph.get_distance(self.current_node, next_node)
        self.current_node = next_node
        self.unvisited_nodes.remove(next_node)

    def complete_path(self):
        while self.unvisited_nodes:
            self.move()
        # Return to start
        self.total_distance += self.graph.get_distance(self.current_node, self.path[0])
        self.path.append(self.path[0])

class ACO:
    def __init__(self, graph, num_ants, num_iterations, decay=0.5, alpha=1.0, beta=3.0, seed=None):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.best_distance_history = []

    def run(self):
        # Set random seed at the beginning of each run
        if self.seed is not None:
            np.random.seed(self.seed)
            
        best_path = None
        best_distance = np.inf

        for _ in range(self.num_iterations):
            ants = [Ant(self.graph, self.alpha, self.beta) for _ in range(self.num_ants)]
            for ant in ants:
                ant.complete_path()

                if ant.total_distance < best_distance:
                    best_path = ant.path
                    best_distance = ant.total_distance

            self.update_pheromones(ants)
            self.best_distance_history.append(best_distance)
        return best_path, best_distance

    def update_pheromones(self, ants):
        # Decay pheromones
        for node1, node2 in self.graph.graph.edges():
            current_pheromone = self.graph.get_pheromone(node1, node2)
            self.graph.set_pheromone(node1, node2, current_pheromone * self.decay)

        # Add new pheromones
        for ant in ants:
            pheromone_contribution = self.alpha / ant.total_distance

            for i in range(len(ant.path) - 1):
                from_node = ant.path[i]
                to_node = ant.path[i + 1]
                
                current_pheromone = self.graph.get_pheromone(from_node, to_node)
                new_pheromone = current_pheromone + pheromone_contribution
                self.graph.set_pheromone(from_node, to_node, new_pheromone)


    


