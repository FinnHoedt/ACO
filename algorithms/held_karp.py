import numpy as np
from itertools import combinations

def held_karp(dist_matrix):
    """
    Held-Karp algorithm for exact TSP solution using dynamic programming.
    
    Time complexity: O(n^2 * 2^n)
    Space complexity: O(n * 2^n)
    
    Args:
        dist_matrix: 2D array representing distances between cities
        
    Returns:
        tuple: (optimal_cost, optimal_path)
    """
    n = len(dist_matrix)
    
    if n <= 1:
        return 0, [0]
    if n == 2:
        return dist_matrix[0][1] + dist_matrix[1][0], [0, 1, 0]
    
    C = {}

    for k in range(1, n):
        C[(1 << k, k)] = (dist_matrix[0][k], [0, k])


    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
        
            bits = sum([1 << k for k in subset])
            
      
            for k in subset:
                prev_bits = bits & ~(1 << k)  
                res = []
                
             
                for m in subset:
                    if m == k:
                        continue
                    prev_cost, prev_path = C[(prev_bits, m)]
                    cost = prev_cost + dist_matrix[m][k]
                    res.append((cost, prev_path + [k]))
                
                C[(bits, k)] = min(res)

  
    bits = (1 << n) - 2 
    res = []
    for k in range(1, n):
        cost, path = C[(bits, k)]
        cost += dist_matrix[k][0]  
        res.append((cost, path + [0]))
    
    return min(res)

def solve_tsp_exact(graph):
    """
    Solve TSP exactly using Held-Karp algorithm.
    
    Args:
        graph: Graph object with distance matrix
        
    Returns:
        tuple: (optimal_path, optimal_distance)
    """
    dist_matrix = graph.distances
    optimal_distance, optimal_path = held_karp(dist_matrix)
    return optimal_path, optimal_distance 