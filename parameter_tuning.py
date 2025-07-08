import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import product
from helper.graph import generate_random_graph, Graph
from algorithms.aco import ACO
from algorithms.held_karp import solve_tsp_exact
import pandas as pd
import seaborn as sns

class ParameterTuner:
    def __init__(self, graph=None, num_runs=5, verbose=True, num_graph_instances=3):
        """
        Initialize parameter tuner for ACO algorithm
        
        Args:
            graph: Graph object to optimize (if None, will generate multiple instances)
            num_runs: Number of runs per parameter combination per graph instance
            verbose: Whether to print progress information
            num_graph_instances: Number of different graph instances to test on
        """
        self.graph = graph
        self.num_runs = num_runs
        self.verbose = verbose
        self.num_graph_instances = num_graph_instances
        self.results = []
        
    def tune_parameters(self, parameter_ranges=None, num_ants=None, num_iterations=100, 
                       num_nodes=10, base_seed=42):
        """
        Tune ACO parameters using grid search on multiple graph instances
        
        Args:
            parameter_ranges: Dict with parameter ranges, if None uses default ranges
            num_ants: Fixed number of ants (if None, uses number of nodes)
            num_iterations: Fixed number of iterations for ACO runs
            num_nodes: Number of nodes in graphs (used if self.graph is None)
            base_seed: Base seed for graph generation (will be varied)
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'decay': [0.1, 0.3, 0.5, 0.7, 0.9],
                'alpha': [0.5, 1.0, 1.5, 2.0, 2.5],
                'beta': [1.0, 2.0, 3.0, 4.0, 5.0]
            }
        
        if self.graph is None:
            print(f"Generating {self.num_graph_instances} different {num_nodes}-node graphs...")
            self.graphs = []
            for i in range(self.num_graph_instances):
                distances, coordinates = generate_random_graph(num_nodes, base_seed + i)
                graph = Graph(distances)
                self.graphs.append(graph)
        else:
            self.graphs = [self.graph]
            self.num_graph_instances = 1
        
        if num_ants is None:
            num_ants = num_nodes if self.graph is None else self.graph.num_nodes
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        total_runs = total_combinations * self.num_runs * self.num_graph_instances
        
        print(f"Testing {total_combinations} parameter combinations")
        print(f"On {self.num_graph_instances} graph instances with {self.num_runs} runs each")
        print(f"Total ACO runs: {total_runs}")
        
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            
            if self.verbose:
                print(f"\nCombination {i+1}/{total_combinations}: {param_dict}")
            
            all_distances = []
            all_times = []
            
            for graph_idx, graph in enumerate(self.graphs):
                if self.verbose and self.num_graph_instances > 1:
                    print(f"  Graph {graph_idx+1}/{self.num_graph_instances}")
                
                for run in range(self.num_runs):
                    start_time = time.time()
                    
                    graph.reset_pheromones()
                    
                    run_seed = base_seed + graph_idx * 1000 + run
                    
                    aco = ACO(graph, num_ants=num_ants, num_iterations=num_iterations, 
                             seed=run_seed, **param_dict)
                    path, distance = aco.run()
                    
                    run_time = time.time() - start_time
                    all_distances.append(distance)
                    all_times.append(run_time)
            
            result = {
                'num_ants': num_ants,
                'num_iterations': num_iterations,
                'num_graph_instances': self.num_graph_instances,
                **param_dict,
                'mean_distance': np.mean(all_distances),
                'std_distance': np.std(all_distances),
                'min_distance': np.min(all_distances),
                'max_distance': np.max(all_distances),
                'mean_time': np.mean(all_times),
                'std_time': np.std(all_times),
                'cv_distance': np.std(all_distances) / np.mean(all_distances) * 100,  # Coefficient of variation
                'success_rate': 1.0
            }
            
            self.results.append(result)
            
            if self.verbose:
                print(f"  Mean distance: {result['mean_distance']:.2f} (±{result['std_distance']:.2f})")
                print(f"  Best distance: {result['min_distance']:.2f}")
                print(f"  CV: {result['cv_distance']:.1f}%")
                print(f"  Mean time: {result['mean_time']:.3f}s")
        
        return self.results
    
    def get_best_parameters(self, top_n=5, criterion='mean_distance'):
        """
        Get the best parameter combinations
        
        Args:
            top_n: Number of best combinations to return
            criterion: Metric to optimize ('mean_distance', 'min_distance', 'mean_time')
        """
        if not self.results:
            raise ValueError("No results available. Run tune_parameters() first.")
        
        reverse = criterion == 'success_rate'
        sorted_results = sorted(self.results, key=lambda x: x[criterion], reverse=reverse)
        
        return sorted_results[:top_n]
    
    def display_results(self, top_n=10):
        """Display the best parameter combinations"""
        if not self.results:
            print("No results available. Run tune_parameters() first.")
            return
        
        print("\n" + "="*80)
        print("PARAMETER TUNING RESULTS")
        print("="*80)
        
        best_params = self.get_best_parameters(top_n, 'mean_distance')
        
        print(f"\nTop {top_n} parameter combinations (by mean distance):")
        if self.num_graph_instances > 1:
            print(f"Results averaged across {self.num_graph_instances} different graph instances")
        print("-" * 80)
        
        for i, result in enumerate(best_params, 1):
            print(f"\nRank {i}:")
            print(f"  Parameters: ants={result['num_ants']}, iterations={result['num_iterations']}, "
                  f"decay={result['decay']}, alpha={result['alpha']}, beta={result['beta']}")
            print(f"  Mean distance: {result['mean_distance']:.2f} (±{result['std_distance']:.2f})")
            print(f"  Best distance: {result['min_distance']:.2f}")
            print(f"  Coefficient of variation: {result['cv_distance']:.1f}%")
            print(f"  Mean time: {result['mean_time']:.3f}s")
            print(f"  Range: {result['min_distance']:.2f} - {result['max_distance']:.2f}")
    
    def plot_parameter_analysis(self):
        """Create visualizations of parameter effects"""
        if not self.results:
            print("No results available. Run tune_parameters() first.")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Parameter Analysis', fontsize=16)
        
        parameters = ['decay', 'alpha', 'beta']
        
        for i, param in enumerate(parameters):
            ax = axes[i]
            
            param_groups = df.groupby(param)['mean_distance'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(param_groups[param], param_groups['mean'], 
                       yerr=param_groups['std'], marker='o', capsize=5)
            ax.set_xlabel(param)
            ax.set_ylabel('Mean Distance')
            ax.set_title(f'Effect of {param}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(8, 6))
        correlation_params = ['decay', 'alpha', 'beta', 'mean_distance', 'mean_time']
        corr_matrix = df[correlation_params].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def compare_with_exact(self, best_params=None):
        """Compare the best parameters with exact solution"""
        if best_params is None:
            if not self.results:
                print("No results available. Run tune_parameters() first.")
                return
            best_params = self.get_best_parameters(1)[0]
        
        print("\n" + "="*60)
        print("COMPARISON WITH EXACT SOLUTION")
        print("="*60)
        
        self.graph.reset_pheromones()
        
        print("Running ACO with best parameters...")
        aco_start = time.time()
        aco = ACO(self.graph, 
                  num_ants=best_params['num_ants'],
                  num_iterations=best_params['num_iterations'],
                  decay=best_params['decay'],
                  alpha=best_params['alpha'],
                  beta=best_params['beta'])
        
        aco_path, aco_distance = aco.run()
        aco_time = time.time() - aco_start
        
        print("Calculating exact solution...")
        exact_start = time.time()
        exact_path, exact_distance = solve_tsp_exact(self.graph)
        exact_time = time.time() - exact_start
        
        gap = ((aco_distance - exact_distance) / exact_distance) * 100
        
        print(f"\nBest parameters found:")
        print(f"  Ants: {best_params['num_ants']}")
        print(f"  Iterations: {best_params['num_iterations']}")
        print(f"  Decay: {best_params['decay']}")
        print(f"  Alpha: {best_params['alpha']}")
        print(f"  Beta: {best_params['beta']}")
        
        print(f"\nResults:")
        print(f"  ACO distance: {aco_distance:.2f}")
        print(f"  Exact distance: {exact_distance:.2f}")
        print(f"  Gap: {gap:.2f}%")
        print(f"  ACO time: {aco_time:.3f}s")
        print(f"  Exact time: {exact_time:.3f}s")
        print(f"  Speedup: {exact_time/aco_time:.1f}x")
        
        return {
            'aco_distance': aco_distance,
            'exact_distance': exact_distance,
            'gap': gap,
            'aco_time': aco_time,
            'exact_time': exact_time
        }


def tune_aco_parameters(nodes=15, seed=42, parameter_ranges=None, num_runs=3, 
                       num_ants=None, num_iterations=100, compare_exact=False,
                       num_graph_instances=5):
    """
    Convenience function to tune ACO parameters for multiple graph instances
    
    Args:
        nodes: Number of nodes in the graphs
        seed: Base random seed for reproducibility
        parameter_ranges: Custom parameter ranges (optional)
        num_runs: Number of runs per parameter combination per graph instance
        num_ants: Fixed number of ants (if None, uses number of nodes)
        num_iterations: Fixed number of iterations for ACO runs
        compare_exact: Whether to compare with exact solution (only for small graphs)
        num_graph_instances: Number of different graph instances to test on
    """
    print(f"Setting up parameter tuning for {nodes}-node graphs...")
    
    tuner = ParameterTuner(graph=None, num_runs=num_runs, num_graph_instances=num_graph_instances)
    
    print("Starting parameter tuning...")
    start_time = time.time()
    results = tuner.tune_parameters(parameter_ranges, num_ants, num_iterations, 
                                   num_nodes=nodes, base_seed=seed)
    total_time = time.time() - start_time
    
    print(f"\nParameter tuning completed in {total_time:.2f} seconds")
    
    tuner.display_results()
    
    tuner.plot_parameter_analysis()
    
    if compare_exact and nodes <= 12:
        print("\nComparing best parameters with exact solution on first graph instance...")
        comparison_tuner = ParameterTuner(graph=tuner.graphs[0], num_runs=1)
        comparison_tuner.results = results 
        comparison_tuner.compare_with_exact()
    elif compare_exact:
        print(f"\nSkipping exact comparison for {nodes} nodes (too large).")
    
    return tuner


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ACO Parameter Tuning')
    parser.add_argument('--nodes', type=int, default=15, help='Number of nodes (default: 15)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per parameter combination per graph')
    parser.add_argument('--ants', type=int, default=None, help='Number of ants (default: number of nodes)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations (default: 100)')
    parser.add_argument('--instances', type=int, default=5, help='Number of graph instances (default: 5)')
    parser.add_argument('--compare-exact', action='store_true', help='Compare with exact solution')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with fewer parameters')
    
    args = parser.parse_args()
    
    if args.nodes < 3:
        print("Error: Minimum 3 nodes required")
        exit(1)
    
    if args.quick_test:
        print("Running quick test with reduced parameter space...")
        quick_ranges = {
            'decay': [0.1, 0.5, 0.9],
            'alpha': [0.5, 1.0, 2.0],
            'beta': [1.0, 3.0, 5.0]
        }
        tuner = tune_aco_parameters(args.nodes, args.seed, quick_ranges, 
                                   num_runs=args.runs, num_ants=args.ants, 
                                   num_iterations=args.iterations,
                                   compare_exact=args.compare_exact,
                                   num_graph_instances=args.instances)
    else:
        tuner = tune_aco_parameters(args.nodes, args.seed, num_runs=args.runs, 
                                   num_ants=args.ants, num_iterations=args.iterations,
                                   compare_exact=args.compare_exact,
                                   num_graph_instances=args.instances) 