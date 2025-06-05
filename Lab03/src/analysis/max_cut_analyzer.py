import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, kruskal
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class MaxCutAnalyzer:
    """
    Comprehensive statistical analysis tool for max-cut solver performance comparison.
    """
    
    def __init__(self, csv_file_path):
        """Initialize with CSV file path."""
        self.df = pd.read_csv(csv_file_path)
        self.solvers = self.df['solver'].unique()
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and clean the data for analysis."""
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
            'p_matrix': p_matrix
        }

# Usage example:
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual CSV file path
    analyzer = MaxCutAnalyzer('your_file.csv')
    results = analyzer.run_complete_analysis()
    
    # Save results to files
    if 'pairwise_tests' in results and len(results['pairwise_tests']) > 0:
        results['pairwise_tests'].to_csv('pairwise_test_results.csv', index=False)
    
    results['one_vs_all'].to_csv('one_vs_all_results.csv', index=False)
    
    print("\nAnalysis complete! Results saved to CSV files and visualization saved as 'maxcut_analysis.png'")