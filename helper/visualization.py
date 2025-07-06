import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(graph, aco_path, coordinates, aco_distance, exact_path=None, exact_distance=None):
    """Visualize the graph and highlight the paths found by different algorithms"""
    
    # Determine number of subplots based on whether we have exact solution
    if exact_path is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    pos = {i: coordinates[i] for i in range(len(coordinates))}
    
    # Plot 1: Full graph
    nx.draw(graph.graph, pos, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=12, font_weight='bold')
    ax1.set_title("Complete Graph")
    ax1.axis('equal')
    
    # Plot 2: ACO path
    aco_edges = [(aco_path[i], aco_path[i+1]) for i in range(len(aco_path)-1)]
    
    # Draw all nodes
    nx.draw_networkx_nodes(graph.graph, pos, ax=ax2, node_color='lightblue', 
                          node_size=500)
    nx.draw_networkx_labels(graph.graph, pos, ax=ax2, font_size=12, font_weight='bold')
    
    # Draw ACO path edges in red
    nx.draw_networkx_edges(graph.graph, pos, edgelist=aco_edges, ax=ax2,
                          edge_color='red', width=3, label='ACO Path')
    
    # Draw other edges in light gray
    other_edges = [edge for edge in graph.graph.edges() if edge not in aco_edges and (edge[1], edge[0]) not in aco_edges]
    nx.draw_networkx_edges(graph.graph, pos, edgelist=other_edges, ax=ax2,
                          edge_color='lightgray', width=1, alpha=0.5)
    
    ax2.set_title(f"ACO Path (Distance: {aco_distance:.2f})")
    ax2.axis('equal')
    
    # Plot 3: Exact path (if available)
    if exact_path is not None:
        exact_edges = [(exact_path[i], exact_path[i+1]) for i in range(len(exact_path)-1)]
        
        # Draw all nodes
        nx.draw_networkx_nodes(graph.graph, pos, ax=ax3, node_color='lightblue', 
                              node_size=500)
        nx.draw_networkx_labels(graph.graph, pos, ax=ax3, font_size=12, font_weight='bold')
        
        # Draw exact path edges in green
        nx.draw_networkx_edges(graph.graph, pos, edgelist=exact_edges, ax=ax3,
                              edge_color='green', width=3, label='Optimal Path')
        
        # Draw other edges in light gray
        other_edges_exact = [edge for edge in graph.graph.edges() if edge not in exact_edges and (edge[1], edge[0]) not in exact_edges]
        nx.draw_networkx_edges(graph.graph, pos, edgelist=other_edges_exact, ax=ax3,
                              edge_color='lightgray', width=1, alpha=0.5)
        
        ax3.set_title(f"Optimal Path (Distance: {exact_distance:.2f})")
        ax3.axis('equal')
        
        # Add comparison text
        gap = ((aco_distance - exact_distance) / exact_distance) * 100
        plt.figtext(0.5, 0.02, f"Gap: {gap:.2f}% | ACO: {aco_distance:.2f} | Optimal: {exact_distance:.2f}", 
                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()