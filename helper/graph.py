import networkx as nx
import numpy as np

class Graph:
    def __init__(self, distances):
        self.distances = distances
        self.num_nodes = len(distances)
        self.graph = nx.Graph()
        
        for i in range(self.num_nodes):
            self.graph.add_node(i)
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if distances[i][j] > 0:
                    self.graph.add_edge(i, j, 
                                      distance=distances[i][j], 
                                      pheromone=1.0)
    
    def get_distance(self, node1, node2):
        """Get distance between two nodes"""
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]['distance']
        return float('inf')
    
    def get_pheromone(self, node1, node2):
        """Get pheromone level between two nodes"""
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]['pheromone']
        return 0.0
    
    def set_pheromone(self, node1, node2, value):
        """Set pheromone level between two nodes"""
        if self.graph.has_edge(node1, node2):
            self.graph[node1][node2]['pheromone'] = value
    
    def get_neighbors(self, node):
        """Get all neighbors of a node"""
        return list(self.graph.neighbors(node))
    
    def reset_pheromones(self, initial_value=1.0):
        """Reset all pheromone levels to initial value"""
        for node1, node2 in self.graph.edges():
            self.graph[node1][node2]['pheromone'] = initial_value
    

def generate_random_graph(num_nodes, seed=42):
    """Generate a random symmetric distance matrix for TSP"""
    np.random.seed(seed)
    
    coordinates = np.random.rand(num_nodes, 2) * 100
    
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    return distances, coordinates