import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from joblib import Parallel, delayed
import copy

class Ant:
    def __init__(self, graph):
        self.graph = graph
        self.current_node = np.random.randint(graph.num_nodes)
        self.path = [self.current_node]
        self.total_distance = 0
        self.unvisited_nodes = set(range(self.graph.num_nodes))
        self.unvisited_nodes.remove(self.current_node)

    def select_next_node(self):
        probabilities = np.zeros(self.graph.num_nodes)

        for node in self.unvisited_nodes:
            distance = self.graph.get_distance(self.current_node, node)
            if distance > 0:
                pheromone = self.graph.get_pheromone(self.current_node, node)
                probabilities[node] = (pheromone ** 2 / distance)

        if probabilities.sum() == 0:
            # If no valid moves, select randomly from unvisited nodes
            return np.random.choice(list(self.unvisited_nodes))
        
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

def run_single_ant(graph_copy):
    """Helper function to run a single ant - needed for multiprocessing"""
    ant = Ant(graph_copy)
    ant.complete_path()
    return ant


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
            
            # Extract matrices for parallel processing (more memory efficient)
            pheromone_matrix, distance_matrix = self._extract_matrices()
            num_nodes = self.graph.num_nodes
            
            # Run ants in parallel using optimized approach
            ants = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(run_optimized_ant)(pheromone_matrix, distance_matrix, num_nodes) 
                for _ in range(self.num_ants)
            )
            
            # Find the best ant in this iteration
            iteration_best_distance = best_distance
            for ant_data in ants:
                path, total_distance = ant_data
                if total_distance < best_distance:
                    best_path = path.copy()
                    best_distance = total_distance

            # Update pheromones using the original graph
            self.update_pheromones_optimized(ants)
            self.best_distance_history.append(best_distance)
            
            if best_distance < iteration_best_distance:
                print(f"New best distance found: {best_distance:.2f}")

        return best_path, best_distance

    def _extract_matrices(self):
        """Extract pheromone and distance matrices from the NetworkX graph"""
        num_nodes = self.graph.num_nodes
        pheromone_matrix = np.zeros((num_nodes, num_nodes))
        distance_matrix = np.full((num_nodes, num_nodes), np.inf)
        
        # Fill diagonal with zeros (distance from node to itself)
        np.fill_diagonal(distance_matrix, 0)
        
        # Extract from NetworkX graph
        for edge in self.graph.graph.edges():
            node1, node2 = edge
            distance = self.graph.get_distance(node1, node2)
            pheromone = self.graph.get_pheromone(node1, node2)
            
            # Make symmetric matrices
            distance_matrix[node1, node2] = distance
            distance_matrix[node2, node1] = distance
            pheromone_matrix[node1, node2] = pheromone
            pheromone_matrix[node2, node1] = pheromone
        
        return pheromone_matrix, distance_matrix

    def update_pheromones_optimized(self, ant_data_list):
        # Decay pheromones
        for edge in self.graph.graph.edges():
            node1, node2 = edge
            current_pheromone = self.graph.get_pheromone(node1, node2)
            self.graph.set_pheromone(node1, node2, current_pheromone * self.decay)
        
        # Add new pheromones
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
    
    # Return to start
    total_distance += distance_matrix[current_node, path[0]]
    path.append(path[0])
    
    return path, total_distance

# Example usage function
def compare_performance():
    """
    Example function to compare performance between sequential and parallel versions.
    """
    from helper.graph import Graph, generate_random_graph  # Assuming you have a Graph class
    import time
    
    distances, coordinates = generate_random_graph(20, 42)
    
    print("Creating graph...")
    graph = Graph(distances)
    
    
    print("Running sequential ACO...")
    start_time = time.time()
    from algorithms.aco import ACO
    sequential_aco = ACO(graph, 20, 100)
    seq_path, seq_distance = sequential_aco.run()
    seq_time = time.time() - start_time
    
    print("\nRunning optimized parallel ACO...")
    start_time = time.time()
    parallel_aco = ACOParallel(graph, 20, 100)
    parallel_path, parallel_distance = parallel_aco.run()
    parallel_time = time.time() - start_time

    print(f"Sequential: {seq_time:.2f} seconds, distance: {seq_distance:.2f}")

    
    print(f"Parallel: {parallel_time:.2f} seconds, distance: {parallel_distance:.2f}")
    print(f"Parallel speedup: {seq_time/parallel_time:.2f}x")
    print(f"Parallel convergence: {seq_distance/parallel_distance:.2f}x")

if __name__ == "__main__":
    compare_performance() 