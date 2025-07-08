import numpy as np
from joblib import Parallel, delayed


class ACOParallel:
    """
    More memory-efficient version that avoids deep copying the entire graph.
    Instead, it extracts only the necessary matrices for parallel processing.
    """
    def __init__(self, graph, num_ants, num_iterations, decay=0.5, alpha=1.0, n_jobs=-1):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.best_distance_history = []

    def run(self):
        best_path = None
        best_distance = np.inf

        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            
            pheromone_matrix, distance_matrix = self._extract_matrices()
            num_nodes = self.graph.num_nodes
            
            ants = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(run_optimized_ant)(pheromone_matrix, distance_matrix, num_nodes) 
                for _ in range(self.num_ants)
            )
            
            for ant_data in ants:
                path, total_distance = ant_data
                if total_distance < best_distance:
                    best_path = path.copy()
                    best_distance = total_distance

            self.update_pheromones_optimized(ants)
            self.best_distance_history.append(best_distance)
            

        return best_path, best_distance

    def _extract_matrices(self):
        """Extract pheromone and distance matrices from the NetworkX graph"""
        num_nodes = self.graph.num_nodes
        pheromone_matrix = np.zeros((num_nodes, num_nodes))
        distance_matrix = np.full((num_nodes, num_nodes), np.inf)
        
        np.fill_diagonal(distance_matrix, 0)
        
        for edge in self.graph.graph.edges():
            node1, node2 = edge
            distance = self.graph.get_distance(node1, node2)
            pheromone = self.graph.get_pheromone(node1, node2)
            
            distance_matrix[node1, node2] = distance
            distance_matrix[node2, node1] = distance
            pheromone_matrix[node1, node2] = pheromone
            pheromone_matrix[node2, node1] = pheromone
        
        return pheromone_matrix, distance_matrix

    def update_pheromones_optimized(self, ant_data_list):
        for edge in self.graph.graph.edges():
            node1, node2 = edge
            current_pheromone = self.graph.get_pheromone(node1, node2)
            self.graph.set_pheromone(node1, node2, current_pheromone * self.decay)
        
        for path, total_distance in ant_data_list:
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                
                current_pheromone = self.graph.get_pheromone(from_node, to_node)
                new_pheromone = current_pheromone + (self.alpha / total_distance)
                self.graph.set_pheromone(from_node, to_node, new_pheromone)

def run_optimized_ant(pheromone_matrix, distance_matrix, num_nodes):
    """
    Optimized ant function that works with numpy matrices directly.
    Returns (path, total_distance) tuple instead of Ant object.
    """
    current_node = np.random.randint(num_nodes)
    path = [current_node]
    total_distance = 0
    unvisited_nodes = set(range(num_nodes))
    unvisited_nodes.remove(current_node)
    
    while unvisited_nodes:
        probabilities = np.zeros(num_nodes)
        
        for node in unvisited_nodes:
            distance = distance_matrix[current_node, node]
            if distance > 0:
                pheromone = pheromone_matrix[current_node, node]
                probabilities[node] = (pheromone ** 2 / distance)
        
        if probabilities.sum() == 0:
            next_node = np.random.choice(list(unvisited_nodes))
        else:
            probabilities /= probabilities.sum()
            next_node = np.random.choice(range(num_nodes), p=probabilities)
        
        path.append(next_node)
        total_distance += distance_matrix[current_node, next_node]
        current_node = next_node
        unvisited_nodes.remove(next_node)
    
    total_distance += distance_matrix[current_node, path[0]]
    path.append(path[0])
    
    return path, total_distance

 