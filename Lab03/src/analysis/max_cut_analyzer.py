import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, pearsonr, spearmanr
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.patches as patches
from itertools import combinations
import warnings

# Suppress specific warnings that are not critical for analysis
# Common warnings include:
# - RuntimeWarnings from log(0) or division by zero in statistical calculations
# - FutureWarnings from pandas/numpy version compatibility
# - UserWarnings from matplotlib about tight_layout
# - Convergence warnings from sklearn when fitting on small datasets
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*ill-conditioned.*')  # sklearn numerical warnings

class MaxCutAnalyzer:    
    def __init__(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path)
        self.solvers = self.df['solver'].unique()
        self.prepare_data()
        
    def prepare_data(self):
        # Convert time to milliseconds for better readability
        self.df['time_ms'] = self.df['time_in_nanoseconds'] / 1e6
        
        # Create efficiency metric (solution quality per unit time)
        self.df['efficiency'] = self.df['max_cut_value'] / (self.df['time_ms'] + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Categorize solvers
        self.exact_solvers = ['Gurobi', 'Brute-force']
        self.approx_solvers = ['QAOA', 'GoemansWilliamson']
        
        print(f"Dataset loaded: {len(self.df)} records")
        print(f"Solvers: {list(self.solvers)}")
        print(f"Graph sizes: {sorted(self.df['node_count'].unique())}")
        
    def descriptive_statistics(self):
        """Generate descriptive statistics for each solver."""
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS")
        print("="*80)
        
        stats_df = self.df.groupby('solver').agg({
            'max_cut_value': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'time_ms': ['mean', 'std', 'median'],
            'efficiency': ['mean', 'std', 'median']
        }).round(4)
        
        print(stats_df)
        return stats_df
    
    def wilcoxon_pairwise_tests(self):
        """Perform pairwise Wilcoxon signed-rank tests for solution quality."""
        print("\n" + "="*80)
        print("PAIRWISE WILCOXON SIGNED-RANK TESTS (Solution Quality)")
        print("="*80)
        
        # Create results matrix
        n_solvers = len(self.solvers)
        p_values = np.ones((n_solvers, n_solvers))
        test_statistics = np.zeros((n_solvers, n_solvers))
        
        results = []
        
        for i, solver1 in enumerate(self.solvers):
            for j, solver2 in enumerate(self.solvers):
                if i != j:
                    # Get data for both solvers on same graphs
                    data1 = self.df[self.df['solver'] == solver1]
                    data2 = self.df[self.df['solver'] == solver2]
                    
                    # Find common graphs
                    common_graphs = set(data1['graph_name']).intersection(set(data2['graph_name']))
                    
                    if len(common_graphs) > 1:
                        # Get paired data
                        values1 = []
                        values2 = []
                        
                        for graph in common_graphs:
                            v1 = data1[data1['graph_name'] == graph]['max_cut_value'].values
                            v2 = data2[data2['graph_name'] == graph]['max_cut_value'].values
                            if len(v1) > 0 and len(v2) > 0:
                                values1.append(v1[0])  # Take first value if multiple runs
                                values2.append(v2[0])
                        
                        if len(values1) > 2:
                            try:
                                stat, p_val = wilcoxon(values1, values2, alternative='two-sided')
                                p_values[i, j] = p_val
                                test_statistics[i, j] = stat
                                
                                effect_size = np.abs(np.mean(values1) - np.mean(values2)) / np.sqrt((np.var(values1) + np.var(values2))/2)
                                
                                results.append({
                                    'Solver 1': solver1,
                                    'Solver 2': solver2,
                                    'N pairs': len(values1),
                                    'Statistic': stat,
                                    'P-value': p_val,
                                    'Significant': p_val < 0.05,
                                    'Effect Size': effect_size
                                })
                                
                                print(f"{solver1} vs {solver2}: n={len(values1)}, stat={stat:.2f}, p={p_val:.4f}, effect_size={effect_size:.3f}")
                            except ValueError as e:
                                print(f"{solver1} vs {solver2}: Cannot perform test - {e}")
                    else:
                        print(f"{solver1} vs {solver2}: Insufficient common graphs ({len(common_graphs)})")
        
        return pd.DataFrame(results), p_values, test_statistics
    
    def one_vs_all_comparisons(self):
        """Perform 1×N comparisons (each solver vs all others combined)."""
        print("\n" + "="*80)
        print("1×N COMPARISONS (Each solver vs all others)")
        print("="*80)
        
        results = []
        
        for solver in self.solvers:
            solver_data = self.df[self.df['solver'] == solver]['max_cut_value']
            others_data = self.df[self.df['solver'] != solver]['max_cut_value']
            
            # Mann-Whitney U test (since groups are independent)
            stat, p_val = mannwhitneyu(solver_data, others_data, alternative='two-sided')
            
            mean_solver = solver_data.mean()
            mean_others = others_data.mean()
            
            results.append({
                'Solver': solver,
                'N_solver': len(solver_data),
                'N_others': len(others_data),
                'Mean_solver': mean_solver,
                'Mean_others': mean_others,
                'Difference': mean_solver - mean_others,
                'U_statistic': stat,
                'P_value': p_val,
                'Significant': p_val < 0.05
            })
            
            print(f"{solver}: Mean={mean_solver:.2f}, Others={mean_others:.2f}, "
                  f"Diff={mean_solver-mean_others:.2f}, p={p_val:.4f}")
        
        return pd.DataFrame(results)
    
    def exact_vs_approximate_analysis(self):
        """Compare exact algorithms vs approximation algorithms."""
        print("\n" + "="*80)
        print("EXACT vs APPROXIMATE ALGORITHMS ANALYSIS")
        print("="*80)
        
        # Filter data for available solvers
        available_exact = [s for s in self.exact_solvers if s in self.solvers]
        available_approx = [s for s in self.approx_solvers if s in self.solvers]
        
        if not available_exact or not available_approx:
            print("Warning: Missing exact or approximate solvers for comparison")
            return None
        
        exact_data = self.df[self.df['solver'].isin(available_exact)]
        approx_data = self.df[self.df['solver'].isin(available_approx)]
        
        # Solution quality comparison
        exact_quality = exact_data['max_cut_value']
        approx_quality = approx_data['max_cut_value']
        
        quality_stat, quality_p = mannwhitneyu(exact_quality, approx_quality, alternative='two-sided')
        
        # Runtime comparison
        exact_time = exact_data['time_ms']
        approx_time = approx_data['time_ms']
        
        time_stat, time_p = mannwhitneyu(exact_time, approx_time, alternative='two-sided')
        
        results = {
            'Quality': {
                'exact_mean': exact_quality.mean(),
                'approx_mean': approx_quality.mean(),
                'statistic': quality_stat,
                'p_value': quality_p,
                'significant': quality_p < 0.05
            },
            'Runtime': {
                'exact_mean': exact_time.mean(),
                'approx_mean': approx_time.mean(),
                'statistic': time_stat,
                'p_value': time_p,
                'significant': time_p < 0.05
            }
        }
        
        print(f"Solution Quality - Exact: {exact_quality.mean():.2f}, Approx: {approx_quality.mean():.2f}, p={quality_p:.4f}")
        print(f"Runtime (ms) - Exact: {exact_time.mean():.2f}, Approx: {approx_time.mean():.2f}, p={time_p:.4f}")
        
        return results
    
    def kruskal_wallis_test(self):
        """Perform Kruskal-Wallis test for overall solver differences."""
        print("\n" + "="*80)
        print("KRUSKAL-WALLIS TEST (Overall solver differences)")
        print("="*80)
        
        # Group data by solver
        solver_groups = [self.df[self.df['solver'] == solver]['max_cut_value'].values 
                        for solver in self.solvers]
        
        stat, p_val = kruskal(*solver_groups)
        
        print(f"Kruskal-Wallis statistic: {stat:.4f}")
        print(f"P-value: {p_val:.4f}")
        print(f"Significant difference between solvers: {p_val < 0.05}")
        
        return stat, p_val
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Solution Quality Box Plot
        plt.subplot(3, 3, 1)
        sns.boxplot(data=self.df, x='solver', y='max_cut_value')
        plt.title('Solution Quality by Solver')
        plt.xticks(rotation=45)
        
        # 2. Runtime Box Plot (log scale)
        plt.subplot(3, 3, 2)
        sns.boxplot(data=self.df, x='solver', y='time_ms')
        plt.yscale('log')
        plt.title('Runtime by Solver (log scale)')
        plt.xticks(rotation=45)
        
        # 3. Efficiency Box Plot
        plt.subplot(3, 3, 3)
        sns.boxplot(data=self.df, x='solver', y='efficiency')
        plt.title('Efficiency by Solver')
        plt.xticks(rotation=45)
        
        # 4. Solution Quality vs Graph Size
        plt.subplot(3, 3, 4)
        for solver in self.solvers:
            solver_data = self.df[self.df['solver'] == solver]
            plt.scatter(solver_data['node_count'], solver_data['max_cut_value'], 
                       label=solver, alpha=0.7)
        plt.xlabel('Node Count')
        plt.ylabel('Max Cut Value')
        plt.title('Solution Quality vs Graph Size')
        plt.legend()
        
        # 5. Runtime vs Graph Size
        plt.subplot(3, 3, 5)
        for solver in self.solvers:
            solver_data = self.df[self.df['solver'] == solver]
            plt.scatter(solver_data['node_count'], solver_data['time_ms'], 
                       label=solver, alpha=0.7)
        plt.xlabel('Node Count')
        plt.ylabel('Runtime (ms)')
        plt.yscale('log')
        plt.title('Runtime vs Graph Size (log scale)')
        plt.legend()
        
        # 6. Quality vs Runtime Scatter
        plt.subplot(3, 3, 6)
        for solver in self.solvers:
            solver_data = self.df[self.df['solver'] == solver]
            plt.scatter(solver_data['time_ms'], solver_data['max_cut_value'], 
                       label=solver, alpha=0.7)
        plt.xlabel('Runtime (ms)')
        plt.ylabel('Max Cut Value')
        plt.xscale('log')
        plt.title('Quality vs Runtime Trade-off')
        plt.legend()
        
        # 7. Violin Plot for Solution Quality
        plt.subplot(3, 3, 7)
        sns.violinplot(data=self.df, x='solver', y='max_cut_value')
        plt.title('Solution Quality Distribution')
        plt.xticks(rotation=45)
        
        # 8. Performance by Graph Size Categories
        plt.subplot(3, 3, 8)
        # Create size categories
        self.df['size_category'] = pd.cut(self.df['node_count'], 
                                         bins=3, labels=['Small', 'Medium', 'Large'])
        sns.boxplot(data=self.df, x='size_category', y='max_cut_value', hue='solver')
        plt.title('Performance by Graph Size Category')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 9. Statistical Significance Heatmap (if we have pairwise results)
        plt.subplot(3, 3, 9)
        try:
            _, p_matrix, _ = self.wilcoxon_pairwise_tests()
            mask = np.triu(np.ones_like(p_matrix))
            sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       xticklabels=self.solvers, yticklabels=self.solvers,
                       mask=mask, cbar_kws={'label': 'P-value'})
            plt.title('Pairwise P-values Heatmap')
        except:
            plt.text(0.5, 0.5, 'Pairwise tests\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Pairwise P-values Heatmap')
        
        plt.tight_layout()
        plt.savefig('maxcut_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def runtime_vs_node_count_analysis(self):
        """Analyze runtime scaling with respect to node count for each solver."""
        print("\n" + "="*80)
        print("RUNTIME vs NODE COUNT SCALING ANALYSIS")
        print("="*80)
        
        results = []
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Runtime vs Node Count Analysis', fontsize=16)
        
        for i, solver in enumerate(self.solvers):
            solver_data = self.df[self.df['solver'] == solver].copy()
            
            if len(solver_data) < 3:
                print(f"Insufficient data for {solver} (n={len(solver_data)})")
                continue
                
            nodes = solver_data['node_count'].values.reshape(-1, 1)
            runtime = solver_data['time_ms'].values
            log_runtime = np.log10(runtime + 1e-6)  # Add small value to avoid log(0)
            
            # Linear correlation
            linear_corr, linear_p = pearsonr(solver_data['node_count'], solver_data['time_ms'])
            
            # Log-log correlation (for power law detection)
            log_nodes = np.log10(solver_data['node_count'])
            loglog_corr, loglog_p = pearsonr(log_nodes, log_runtime)
            
            # Spearman correlation (non-parametric)
            spearman_corr, spearman_p = spearmanr(solver_data['node_count'], solver_data['time_ms'])
            
            # Linear regression
            lr = LinearRegression()
            lr.fit(nodes, runtime)
            linear_r2 = lr.score(nodes, runtime)
            linear_pred = lr.predict(nodes)
            
            results.append({
                'Solver': solver,
                'N_samples': len(solver_data),
                'Linear_correlation': linear_corr,
                'Linear_p_value': linear_p,
                'Spearman_correlation': spearman_corr,
                'Spearman_p_value': spearman_p,
                'Linear_R2': linear_r2
            })
            
            # Plotting
            ax = axes[i//2, i%2] if i < 4 else None
            if ax is not None:
                # Scatter plot
                ax.scatter(solver_data['node_count'], solver_data['time_ms'], 
                          alpha=0.7, label='Data points')
                
                # Fit lines
                sorted_indices = np.argsort(solver_data['node_count'])
                ax.plot(solver_data['node_count'].iloc[sorted_indices], 
                       linear_pred[sorted_indices], 'r--', 
                       label=f'Linear (R²={linear_r2:.3f})')
                
                ax.set_xlabel('Node Count')
                ax.set_ylabel('Runtime (ms)')
                ax.set_title(f'{solver}')
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            print(f"\n{solver}:")
            print(f"Linear correlation: r={linear_corr:.3f}, p={linear_p:.4f}")
            print(f"Spearman correlation: ρ={spearman_corr:.3f}, p={spearman_p:.4f}")
        
        plt.tight_layout()
        plt.savefig('runtime_vs_nodes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pd.DataFrame(results)
    
    def runtime_vs_edge_count_analysis(self):
        """Analyze runtime scaling with respect to edge count for each solver."""
        print("\n" + "="*80)
        print("RUNTIME vs EDGE COUNT SCALING ANALYSIS")
        print("="*80)
        
        results = []
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Runtime vs Edge Count Analysis', fontsize=16)
        
        for i, solver in enumerate(self.solvers):
            solver_data = self.df[self.df['solver'] == solver].copy()
            
            if len(solver_data) < 3:
                print(f"Insufficient data for {solver} (n={len(solver_data)})")
                continue
                
            edges = solver_data['edge_count'].values.reshape(-1, 1)
            runtime = solver_data['time_ms'].values
            log_runtime = np.log10(runtime + 1e-6)
            
            # Linear correlation
            linear_corr, linear_p = pearsonr(solver_data['edge_count'], solver_data['time_ms'])
            
            # Log-log correlation
            log_edges = np.log10(solver_data['edge_count'])
            loglog_corr, loglog_p = pearsonr(log_edges, log_runtime)
            
            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(solver_data['edge_count'], solver_data['time_ms'])
            
            # Linear regression
            lr = LinearRegression()
            lr.fit(edges, runtime)
            linear_r2 = lr.score(edges, runtime)
            linear_pred = lr.predict(edges)
            
            results.append({
                'Solver': solver,
                'N_samples': len(solver_data),
                'Linear_correlation': linear_corr,
                'Linear_p_value': linear_p,
                'Spearman_correlation': spearman_corr,
                'Spearman_p_value': spearman_p,
                'Linear_R2': linear_r2
            })
            
            # Plotting
            ax = axes[i//2, i%2] if i < 4 else None
            if ax is not None:
                # Scatter plot
                ax.scatter(solver_data['edge_count'], solver_data['time_ms'], 
                          alpha=0.7, label='Data points')
                
                # Fit lines
                sorted_indices = np.argsort(solver_data['edge_count'])
                ax.plot(solver_data['edge_count'].iloc[sorted_indices], 
                       linear_pred[sorted_indices], 'r--', 
                       label=f'Linear (R²={linear_r2:.3f})')
                
                ax.set_xlabel('Edge Count')
                ax.set_ylabel('Runtime (ms)')
                ax.set_title(f'{solver}')
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            print(f"\n{solver}:")
            print(f"  Linear correlation: r={linear_corr:.3f}, p={linear_p:.4f}")
            print(f"  Spearman correlation: ρ={spearman_corr:.3f}, p={spearman_p:.4f}")
        
        plt.tight_layout()
        plt.savefig('runtime_vs_edges.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pd.DataFrame(results)
    
    def runtime_heatmap_analysis(self):
        """Create heatmaps showing runtime as a function of node count and edge count."""
        print("\n" + "="*80)
        print("RUNTIME HEATMAP ANALYSIS (Node Count vs Edge Count)")
        print("="*80)
        
        # Create figure with subplots for each solver
        n_solvers = len(self.solvers)
        cols = 2
        rows = (n_solvers + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if n_solvers == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Runtime Heatmaps: Node Count vs Edge Count', fontsize=16)
        
        results = []
        
        for i, solver in enumerate(self.solvers):
            solver_data = self.df[self.df['solver'] == solver].copy()
            
            if len(solver_data) < 3:
                print(f"Insufficient data for {solver} (n={len(solver_data)})")
                continue
            
            # Get the subplot
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Extract data
            nodes = solver_data['node_count'].values
            edges = solver_data['edge_count'].values
            runtime = solver_data['time_ms'].values
            
            # Calculate statistics
            node_range = nodes.max() - nodes.min()
            edge_range = edges.max() - edges.min()
            runtime_range = runtime.max() - runtime.min()
            
            # Calculate correlation with graph density (edges/nodes²)
            max_possible_edges = nodes * (nodes - 1) / 2
            density = edges / max_possible_edges
            density_corr = np.corrcoef(density, runtime)[0, 1] if len(density) > 1 else 0
            
            # Calculate partial correlations
            node_runtime_corr, node_p = pearsonr(nodes, runtime)
            edge_runtime_corr, edge_p = pearsonr(edges, runtime)
            node_edge_corr, _ = pearsonr(nodes, edges)
            
            results.append({
                'Solver': solver,
                'N_samples': len(solver_data),
                'Node_range': f"{nodes.min()}-{nodes.max()}",
                'Edge_range': f"{edges.min()}-{edges.max()}",
                'Runtime_range_ms': f"{runtime.min():.2f}-{runtime.max():.2f}",
                'Node_runtime_correlation': node_runtime_corr,
                'Node_runtime_p_value': node_p,
                'Edge_runtime_correlation': edge_runtime_corr,
                'Edge_runtime_p_value': edge_p,
                'Density_runtime_correlation': density_corr,
                'Node_edge_correlation': node_edge_corr
            })
            
            # Create heatmap
            if len(solver_data) >= 10:  # Need enough points for interpolation
                # Create grid for interpolation
                node_min, node_max = nodes.min(), nodes.max()
                edge_min, edge_max = edges.min(), edges.max()
                
                # Create grid
                grid_nodes = np.linspace(node_min, node_max, 50)
                grid_edges = np.linspace(edge_min, edge_max, 50)
                grid_n, grid_e = np.meshgrid(grid_nodes, grid_edges)
                
                # Interpolate runtime values
                try:
                    grid_runtime = griddata((nodes, edges), np.log10(runtime + 1e-6), 
                                          (grid_n, grid_e), method='cubic', fill_value=np.nan)
                    
                    # Create heatmap
                    im = ax.imshow(grid_runtime, extent=[node_min, node_max, edge_min, edge_max], 
                                  origin='lower', aspect='auto', cmap='viridis')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Log₁₀(Runtime ms)', rotation=270, labelpad=20)
                    
                    # Add contour lines
                    contours = ax.contour(grid_n, grid_e, grid_runtime, levels=8, colors='white', alpha=0.6, linewidths=0.8)
                    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
                    
                except Exception as e:
                    print(f"Could not create interpolated heatmap for {solver}: {e}")
                    # Fallback to scatter plot
                    scatter = ax.scatter(nodes, edges, c=np.log10(runtime + 1e-6), 
                                       cmap='viridis', s=50, alpha=0.7)
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Log₁₀(Runtime ms)', rotation=270, labelpad=20)
            else:
                # Use scatter plot for small datasets
                scatter = ax.scatter(nodes, edges, c=np.log10(runtime + 1e-6), 
                                   cmap='viridis', s=50, alpha=0.7)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Log₁₀(Runtime ms)', rotation=270, labelpad=20)
            
            # Overlay actual data points
            ax.scatter(nodes, edges, c='red', s=20, alpha=0.8, marker='x', linewidths=1)
            
            # Labels and title
            ax.set_xlabel('Node Count')
            ax.set_ylabel('Edge Count')
            ax.set_title(f'{solver}\n(r_nodes={node_runtime_corr:.3f}, r_edges={edge_runtime_corr:.3f})')
            ax.grid(True, alpha=0.3)
            
            print(f"\n{solver}:")
            print(f"  Node-Runtime correlation: r={node_runtime_corr:.3f}, p={node_p:.4f}")
            print(f"  Edge-Runtime correlation: r={edge_runtime_corr:.3f}, p={edge_p:.4f}")
            print(f"  Density-Runtime correlation: r={density_corr:.3f}")
            print(f"  Data points: {len(solver_data)}")
        
        # Hide empty subplots
        total_subplots = rows * cols
        for i in range(n_solvers, total_subplots):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig('runtime_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create combined heatmap for comparison
        self._create_combined_heatmap()
        
        return pd.DataFrame(results)
    
    def _create_combined_heatmap(self):
        """Create a combined heatmap showing all solvers."""
        print("\nCreating combined comparison heatmap...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get overall ranges
        all_nodes = self.df['node_count']
        all_edges = self.df['edge_count']
        
        node_min, node_max = all_nodes.min(), all_nodes.max()
        edge_min, edge_max = all_edges.min(), all_edges.max()
        
        # Create color map for solvers
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.solvers)))
        
        for i, solver in enumerate(self.solvers):
            solver_data = self.df[self.df['solver'] == solver]
            
            if len(solver_data) < 2:
                continue
                
            nodes = solver_data['node_count']
            edges = solver_data['edge_count']
            runtime = solver_data['time_ms']
            
            # Create bubble chart
            # Bubble size proportional to runtime (log scale)
            sizes = 50 + 200 * (np.log10(runtime + 1) / np.log10(runtime.max() + 1))
            
            scatter = ax.scatter(nodes, edges, s=sizes, c=[colors[i]], 
                               alpha=0.6, label=solver, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Node Count')
        ax.set_ylabel('Edge Count')
        ax.set_title('Combined Runtime Analysis\n(Bubble size ∝ Runtime)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        ax.text(0.02, 0.98, 'Larger bubbles = Longer runtime', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('combined_runtime_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete statistical analysis."""
        print("MAX-CUT SOLVER PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Descriptive statistics
        desc_stats = self.descriptive_statistics()
        
        # Statistical tests
        kruskal_stat, kruskal_p = self.kruskal_wallis_test()
        pairwise_results, p_matrix, stat_matrix = self.wilcoxon_pairwise_tests()
        one_vs_all_results = self.one_vs_all_comparisons()
        exact_vs_approx = self.exact_vs_approximate_analysis()
        
        # Scaling analysis
        node_scaling = self.runtime_vs_node_count_analysis()
        edge_scaling = self.runtime_vs_edge_count_analysis()
        heatmap_analysis = self.runtime_heatmap_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        # Summary
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total records analyzed: {len(self.df)}")
        print(f"Solvers compared: {list(self.solvers)}")
        print(f"Overall difference between solvers (Kruskal-Wallis): p={kruskal_p:.4f}")
        
        if len(pairwise_results) > 0:
            sig_pairs = pairwise_results[pairwise_results['Significant']]['Solver 1'].count()
            print(f"Significant pairwise differences: {sig_pairs}/{len(pairwise_results)}")
        
        # Return all results
        return {
            'descriptive_stats': desc_stats,
            'kruskal_wallis': (kruskal_stat, kruskal_p),
            'pairwise_tests': pairwise_results,
            'one_vs_all': one_vs_all_results,
            'exact_vs_approx': exact_vs_approx,
            'p_matrix': p_matrix,
            'node_scaling': node_scaling,
            'edge_scaling': edge_scaling,
            'heatmap_analysis': heatmap_analysis
        }