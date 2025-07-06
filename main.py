import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import time
from helper.graph import generate_random_graph, Graph
from algorithms.aco import ACO
from helper.visualization import visualize_graph
from algorithms.held_karp import solve_tsp_exact

def main():
    parser = argparse.ArgumentParser(description='ACO Algorithmus für TSP')
    parser.add_argument('nodes', type=int, help='Anzahl Knoten')
    parser.add_argument('--ants', type=int, default=None, help='Anzahl Ameisen (default: Anzahl Knoten)')
    parser.add_argument('--iterations', type=int, default=100, help='Anzahl Iterationen (default: 100)')
    parser.add_argument('--decay', type=float, default=0.5, help='Pheromonabbaurate (default: 0.5)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Pheromonintensitätsfaktor (default: 1.0)')
    parser.add_argument('--beta', type=float, default=3.0, help='Heuristikgewichtung (default: 3.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--compare-exact', action='store_true', help='Compare with exact solution')
    
    args = parser.parse_args()
    
    # Set default number of ants to number of nodes if not specified
    if args.ants is None:
        args.ants = args.nodes
    
    if args.nodes < 3:
        print("Error: Mindestens 3 Knoten")
        exit(1)
    
    distances, coordinates = generate_random_graph(args.nodes, args.seed)
    
    graph = Graph(distances)
    
    print(f"ACO läuft mit {args.ants} Ameisen für {args.iterations} Iterationen...")
    aco_start_time = time.time()
    aco = ACO(graph, num_ants=args.ants, num_iterations=args.iterations, 
              decay=args.decay, alpha=args.alpha)
    
    aco_path, aco_distance = aco.run()
    aco_time = time.time() - aco_start_time
    
    print("\n" + "="*50)
    print("ACO ERGEBNISSE")
    print("="*50)
    print(f"Bester gefundener Pfad: {' -> '.join(map(str, aco_path))}")
    print(f"Beste Distanz: {aco_distance:.2f}")
    print(f"Pfadlänge: {len(aco_path) - 1} Kanten")
    print(f"Rechenzeit: {aco_time:.3f} Sekunden")
    
    # Show convergence
    print(f"\nKonvergenz:")
    print(f"Anfängliche beste Distanz: {aco.best_distance_history[0]:.2f}")
    print(f"Finale beste Distanz: {aco.best_distance_history[-1]:.2f}")
    print(f"Verbesserung: {((aco.best_distance_history[0] - aco.best_distance_history[-1]) / aco.best_distance_history[0] * 100):.2f}%")
    
    # Initialize exact solution variables
    exact_path = None
    exact_distance = None
    
    # Compare with exact solution if requested
    if args.compare_exact:
        if args.nodes > 12:
            print(f"\nWarnung: Berechnung für {args.nodes} dauert länger!")
            response = input("Trotzdem fortfahren? (j/N): ")
            if response.lower() != 'j':
                print("Überspringe Berechnung der exakten Lösung.")
                args.compare_exact = False
        
        if args.compare_exact:
            print(f"\nBerechne exakte Lösung mit Held-Karp-Algorithmus...")
            exact_start_time = time.time()
            exact_path, exact_distance = solve_tsp_exact(graph)
            exact_time = time.time() - exact_start_time
            
            print("\n" + "="*50)
            print("EXAKTE LÖSUNG ERGEBNISSE")
            print("="*50)
            print(f"Optimaler Pfad: {' -> '.join(map(str, exact_path))}")
            print(f"Optimale Distanz: {exact_distance:.2f}")
            print(f"Rechenzeit: {exact_time:.3f} Sekunden")
            
            print("\n" + "="*50)
            print("VERGLEICH")
            print("="*50)
            gap = ((aco_distance - exact_distance) / exact_distance) * 100
            print(f"ACO Distanz: {aco_distance:.2f}")
            print(f"Optimale Distanz: {exact_distance:.2f}")
            print(f"Abweichung: {gap:.2f}% {'(ACO hat Optimum gefunden!)' if gap < 0.01 else ''}")
            print(f"ACO Zeit: {aco_time:.3f}s")
            print(f"Exakte Zeit: {exact_time:.3f}s")
            print(f"Beschleunigung: {exact_time/aco_time:.1f}x schneller (ACO)")
    
    print("\nVisualisiere Graph...")
    visualize_graph(graph, aco_path, coordinates, aco_distance, exact_path, exact_distance)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(aco.best_distance_history, 'b-', linewidth=2)
    plt.title('ACO Konvergenz')
    plt.xlabel('Iteration')
    plt.ylabel('Beste gefundene Distanz')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
