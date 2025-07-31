"""
Statistical Analysis Module for Bayesian Optimization Benchmark Results
Performs Friedman and Wilcoxon significance testing on TabPFN vs Vanilla BO results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon
from scipy import stats
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class StatisticalAnalyzer:
   """
   Performs comprehensive statistical analysis on BO benchmark results.
   """
   
   def __init__(self, results_dir: str = "bbob_benchmark_results"):
       """
       Initialize the statistical analyzer.
       
       Args:
           results_dir: Path to the benchmark results directory
       """
       self.results_dir = Path(results_dir)
       self.performance_data = {}
       self.rankings = {}
       self.statistical_results = {}
       
       # Set up matplotlib for publication-quality plots
       plt.style.use('default')
       plt.rcParams.update({
           'font.size': 12,
           'font.family': 'serif',
           'axes.linewidth': 1.2,
           'axes.spines.top': False,
           'axes.spines.right': False,
           'figure.dpi': 100,
           'savefig.dpi': 300,
           'savefig.bbox': 'tight'
       })
       
   def load_dat_file(self, file_path: Path) -> Optional[Dict]:
       """
       Load and process a single .dat file.
       
       Args:
           file_path: Path to the .dat file
           
       Returns:
           Dictionary containing processed data or None if failed
       """
       try:
           # Read the .dat file
           df = pd.read_csv(file_path, delimiter=' ', skipinitialspace=True)
           
           # Extract metadata from file path
           # Path structure: dim_X/algorithm/algorithm_fY_dimX_instZ_repW/data_fY_FunctionName/IOHprofiler_fY_DIMX.dat
           path_parts = file_path.parts
           
           # Find algorithm from path
           algorithm = None
           for part in path_parts:
               if 'vanilla' in part.lower():
                   algorithm = 'vanilla'
                   break
               elif 'tabpfn' in part.lower():
                   algorithm = 'tabpfn'
                   break
           
           if algorithm is None:
               print(f"Warning: Could not determine algorithm from path: {file_path}")
               return None
           
           # Extract function ID, dimension, instance, repetition from filename
           filename = file_path.name  # IOHprofiler_f1_DIM2.dat
           if 'IOHprofiler_f' in filename and '_DIM' in filename:
               try:
                   # Extract function ID
                   f_part = filename.split('_f')[1].split('_')[0]
                   function_id = int(f_part)
                   
                   # Extract dimension
                   dim_part = filename.split('_DIM')[1].split('.')[0]
                   dimension = int(dim_part)
                   
                   # Extract instance and repetition from parent directory name
                   parent_dir = file_path.parent.parent.name  # algorithm_f1_dim2_inst1_rep1
                   if '_inst' in parent_dir and '_rep' in parent_dir:
                       inst_part = parent_dir.split('_inst')[1].split('_rep')[0]
                       rep_part = parent_dir.split('_rep')[1]
                       instance = int(inst_part)
                       repetition = int(rep_part)
                   else:
                       # Default values if not found
                       instance = 1
                       repetition = 1
                       
               except (ValueError, IndexError) as e:
                   print(f"Warning: Could not parse metadata from {filename}: {e}")
                   return None
           else:
               print(f"Warning: Unexpected filename format: {filename}")
               return None
           
           # Process the data
           if 'raw_y_best' not in df.columns:
               print(f"Warning: No 'raw_y_best' column in {file_path}")
               return None
               
           return {
               'algorithm': algorithm,
               'function_id': function_id,
               'dimension': dimension,
               'instance': instance,
               'repetition': repetition,
               'evaluations': df['evaluations'].tolist(),
               'best_so_far': df['raw_y_best'].tolist(),
               'final_best': df['raw_y_best'].iloc[-1],
               'file_path': str(file_path)
           }
           
       except Exception as e:
           print(f"Error processing {file_path}: {e}")
           return None
   
   def collect_all_results(self) -> Dict:
       """
       Collect all .dat files and organize by problem configuration.
       
       Returns:
           Dictionary organized by problem key containing algorithm results
       """
       print(f"Scanning for .dat files in: {self.results_dir}")
       
       # Find all .dat files
       dat_files = list(self.results_dir.rglob("IOHprofiler_f*_DIM*.dat"))
       print(f"Found {len(dat_files)} .dat files")
       
       if len(dat_files) == 0:
           print("Warning: No .dat files found!")
           return {}
       
       # Process each file
       all_data = []
       for dat_file in dat_files:
           data = self.load_dat_file(dat_file)
           if data:
               all_data.append(data)
       
       print(f"Successfully processed {len(all_data)} files")
       
       # Organize data by problem configuration
       organized_data = {}
       
       for data in all_data:
           # Create unique key for each problem configuration
           problem_key = f"f{data['function_id']}_dim{data['dimension']}_inst{data['instance']}"
           
           if problem_key not in organized_data:
               organized_data[problem_key] = {'vanilla': [], 'tabpfn': []}
           
           # Add data to appropriate algorithm list
           algorithm = data['algorithm']
           if algorithm in organized_data[problem_key]:
               organized_data[problem_key][algorithm].append(data)
           else:
               print(f"Warning: Unknown algorithm '{algorithm}' for {problem_key}")
       
       # Filter out problems that don't have both algorithms
       filtered_data = {}
       for problem_key, algo_data in organized_data.items():
           if len(algo_data['vanilla']) > 0 and len(algo_data['tabpfn']) > 0:
               filtered_data[problem_key] = algo_data
           else:
               print(f"Warning: Skipping {problem_key} - missing data for one algorithm")
               print(f"  Vanilla: {len(algo_data['vanilla'])} runs, TabPFN: {len(algo_data['tabpfn'])} runs")
       
       print(f"Final dataset: {len(filtered_data)} problem configurations with both algorithms")
       self.performance_data = filtered_data
       return filtered_data
   
   def calculate_rankings(self, metric: str = 'final_best') -> Dict:
       """
       Calculate statistical rankings for each problem.
       
       Args:
           metric: Which metric to use ('final_best' or 'final_regret')
           
       Returns:
           Dictionary of rankings for each problem
       """
       if not self.performance_data:
           print("Error: No performance data loaded. Run collect_all_results() first.")
           return {}
       
       rankings = {}
       
       for problem_key, algo_data in self.performance_data.items():
           # Calculate median performance for each algorithm
           medians = {}
           
           for algorithm, runs in algo_data.items():
               if metric == 'final_best':
                   values = [run['final_best'] for run in runs]
               elif metric == 'final_regret':
                   # Calculate regret if optimal value is available
                   # For now, use relative performance
                   values = [run['final_best'] for run in runs]
               else:
                   raise ValueError(f"Unknown metric: {metric}")
               
               medians[algorithm] = np.median(values)
           
           # Rank algorithms (1 = best, 2 = worst for minimization)
           sorted_algorithms = sorted(medians.items(), key=lambda x: x[1])
           
           problem_rankings = {}
           for rank, (algorithm, median_val) in enumerate(sorted_algorithms, 1):
               problem_rankings[algorithm] = rank
           
           rankings[problem_key] = problem_rankings
       
       self.rankings = rankings
       return rankings
   
   def perform_friedman_test(self) -> Dict:
       """
       Perform Friedman test to check for significant differences between algorithms.
       Note: For only 2 algorithms, this test is not applicable - use Wilcoxon instead.
       
       Returns:
           Dictionary containing test results
       """
       if not self.rankings:
           print("Error: No rankings calculated. Run calculate_rankings() first.")
           return {}
       
       # Extract ranks for each algorithm across all problems
       vanilla_ranks = []
       tabpfn_ranks = []
       
       for problem_key, problem_rankings in self.rankings.items():
           if 'vanilla' in problem_rankings and 'tabpfn' in problem_rankings:
               vanilla_ranks.append(problem_rankings['vanilla'])
               tabpfn_ranks.append(problem_rankings['tabpfn'])
       
       # Check if we have exactly 2 algorithms
       unique_algorithms = set()
       for problem_rankings in self.rankings.values():
           unique_algorithms.update(problem_rankings.keys())
       
       if len(unique_algorithms) == 2:
           return {
               'test_name': 'Friedman Test',
               'error': 'Friedman test requires 3+ algorithms. For 2 algorithms, use Wilcoxon signed-rank test.',
               'note': 'With exactly 2 algorithms, Wilcoxon signed-rank test is the appropriate choice.',
               'n_algorithms': len(unique_algorithms),
               'algorithms': list(unique_algorithms)
           }
       
       if len(vanilla_ranks) < 3:
           print(f"Warning: Only {len(vanilla_ranks)} problems available. Friedman test needs at least 3.")
           return {'error': 'Insufficient data for Friedman test'}
       
       # Perform Friedman test (only if we have 3+ algorithms)
       try:
           statistic, p_value = friedmanchisquare(vanilla_ranks, tabpfn_ranks)
           
           results = {
               'test_name': 'Friedman Test',
               'statistic': float(statistic),
               'p_value': float(p_value),
               'significant': bool(p_value < 0.05),
               'n_problems': int(len(vanilla_ranks)),
               'vanilla_ranks': [int(r) for r in vanilla_ranks],
               'tabpfn_ranks': [int(r) for r in tabpfn_ranks],
               'interpretation': 'Significant differences exist between algorithms' if p_value < 0.05 
                               else 'No significant differences between algorithms'
           }
           
           return results
           
       except Exception as e:
           print(f"Error in Friedman test: {e}")
           return {'error': str(e)}
   
   def perform_wilcoxon_test(self) -> Dict:
       """
       Perform Wilcoxon signed-rank test for pairwise comparison.
       
       Returns:
           Dictionary containing test results
       """
       if not self.rankings:
           print("Error: No rankings calculated. Run calculate_rankings() first.")
           return {}
       
       # Extract rank differences (vanilla - tabpfn)
       rank_differences = []
       
       for problem_key, problem_rankings in self.rankings.items():
           if 'vanilla' in problem_rankings and 'tabpfn' in problem_rankings:
               diff = problem_rankings['vanilla'] - problem_rankings['tabpfn']
               rank_differences.append(diff)
       
       if len(rank_differences) < 6:
           print(f"Warning: Only {len(rank_differences)} problems available. Wilcoxon test needs at least 6 for reliable results.")
           if len(rank_differences) < 3:
               return {'error': 'Insufficient data for Wilcoxon test (need at least 3 problems)'}
       
       # Remove zero differences (ties)
       non_zero_differences = [d for d in rank_differences if d != 0]
       
       if len(non_zero_differences) == 0:
           return {
               'test_name': 'Wilcoxon Signed-Rank Test',
               'error': 'All rank differences are zero (perfect ties)',
               'n_problems': int(len(rank_differences)),
               'n_ties': int(len(rank_differences)),
               'interpretation': 'Algorithms perform identically (all ties)'
           }
       
       # Perform Wilcoxon signed-rank test
       try:
           statistic, p_value = wilcoxon(non_zero_differences)
           
           # Calculate effect size (median difference)
           median_diff = np.median(rank_differences)
           
           # Interpret results
           if p_value >= 0.05:
               interpretation = "No significant difference between algorithms"
           elif median_diff > 0:
               interpretation = "TabPFN significantly outperforms Vanilla BO"
           else:
               interpretation = "Vanilla BO significantly outperforms TabPFN"
           
           results = {
               'test_name': 'Wilcoxon Signed-Rank Test',
               'statistic': float(statistic),
               'p_value': float(p_value),
               'significant': bool(p_value < 0.05),
               'n_problems': int(len(rank_differences)),
               'n_non_zero': int(len(non_zero_differences)),
               'n_ties': int(len(rank_differences) - len(non_zero_differences)),
               'rank_differences': [int(d) for d in rank_differences],
               'median_difference': float(median_diff),
               'mean_difference': float(np.mean(rank_differences)),
               'interpretation': interpretation
           }
           
           return results
           
       except Exception as e:
           print(f"Error in Wilcoxon test: {e}")
           return {'error': str(e)}
   
   def calculate_summary_statistics(self) -> Dict:
       """
       Calculate overall summary statistics.
       
       Returns:
           Dictionary containing summary statistics
       """
       if not self.rankings:
           print("Error: No rankings calculated. Run calculate_rankings() first.")
           return {}
       
       # Calculate average ranks
       vanilla_ranks = [ranks['vanilla'] for ranks in self.rankings.values() if 'vanilla' in ranks]
       tabpfn_ranks = [ranks['tabpfn'] for ranks in self.rankings.values() if 'tabpfn' in ranks]
       
       # Win/loss statistics
       vanilla_wins = sum(1 for ranks in self.rankings.values() if ranks.get('vanilla', 2) < ranks.get('tabpfn', 1))
       tabpfn_wins = sum(1 for ranks in self.rankings.values() if ranks.get('tabpfn', 2) < ranks.get('vanilla', 1))
       ties = sum(1 for ranks in self.rankings.values() if ranks.get('vanilla', 0) == ranks.get('tabpfn', 0))
       
       # Performance statistics from raw data
       all_vanilla_finals = []
       all_tabpfn_finals = []
       
       for problem_key, algo_data in self.performance_data.items():
           vanilla_finals = [run['final_best'] for run in algo_data['vanilla']]
           tabpfn_finals = [run['final_best'] for run in algo_data['tabpfn']]
           
           all_vanilla_finals.extend(vanilla_finals)
           all_tabpfn_finals.extend(tabpfn_finals)
       
       summary = {
           'n_problems': int(len(self.rankings)),
           'n_total_runs': int(len(all_vanilla_finals) + len(all_tabpfn_finals)),
           'average_ranks': {
               'vanilla': float(np.mean(vanilla_ranks)) if vanilla_ranks else None,
               'tabpfn': float(np.mean(tabpfn_ranks)) if tabpfn_ranks else None
           },
           'rank_std': {
               'vanilla': float(np.std(vanilla_ranks)) if vanilla_ranks else None,
               'tabpfn': float(np.std(tabpfn_ranks)) if tabpfn_ranks else None
           },
           'win_loss_tie': {
               'vanilla_wins': int(vanilla_wins),
               'tabpfn_wins': int(tabpfn_wins),
               'ties': int(ties),
               'vanilla_win_rate': float(vanilla_wins / len(self.rankings)) if self.rankings else 0.0,
               'tabpfn_win_rate': float(tabpfn_wins / len(self.rankings)) if self.rankings else 0.0
           },
           'performance_stats': {
               'vanilla_mean_final': float(np.mean(all_vanilla_finals)) if all_vanilla_finals else None,
               'tabpfn_mean_final': float(np.mean(all_tabpfn_finals)) if all_tabpfn_finals else None,
               'vanilla_std_final': float(np.std(all_vanilla_finals)) if all_vanilla_finals else None,
               'tabpfn_std_final': float(np.std(all_tabpfn_finals)) if all_tabpfn_finals else None
           }
       }
       
       return summary
   
   def create_visualizations(self, save_plots: bool = True, plot_dir: str = "thesis_plots"):
       """
       Create all mandatory visualizations for thesis.
       
       Args:
           save_plots: Whether to save plots to files
           plot_dir: Directory to save plots
       """
       if not self.statistical_results:
           print("Error: No analysis results available. Run run_complete_analysis() first.")
           return
       
       # Create plot directory
       plot_path = Path(plot_dir)
       if save_plots:
           plot_path.mkdir(exist_ok=True)
           print(f"\n📁 Creating thesis visualizations in: {plot_path.absolute()}")
           print(f"   Plots will be saved as both PNG and PDF formats")
       else:
           print(f"\n📊 Creating thesis visualizations (display only)")
       
       # Extract data for plotting
       summary = self.statistical_results.get('summary_statistics', {})
       wlt = summary.get('win_loss_tie', {})
       wilcoxon = self.statistical_results.get('wilcoxon_test', {})
       
       # 1. Win-Loss Bar Chart
       print("\n1. Creating win-loss comparison chart...")
       self._plot_win_loss_chart(wlt, save_plots, plot_dir)
       
       # 2. Rank Difference Distribution
       print("\n2. Creating rank difference distribution...")
       self._plot_rank_differences(wilcoxon, save_plots, plot_dir)
       
       # 3. Performance by Dimension
       print("\n3. Creating performance by dimension chart...")
       self._plot_performance_by_dimension(save_plots, plot_dir)
       
       if save_plots:
           print(f"\n✅ All visualizations created and saved to: {plot_path.absolute()}")
       else:
           print(f"\n✅ All visualizations displayed successfully!")
   
   def _plot_win_loss_chart(self, wlt: Dict, save_plots: bool, plot_dir: str):
       """Create win-loss comparison bar chart."""
       plt.figure(figsize=(10, 7))
       
       algorithms = ['TabPFN BO', 'Vanilla BO']
       wins = [wlt.get('tabpfn_wins', 0), wlt.get('vanilla_wins', 0)]
       ties = wlt.get('ties', 0)
       total_problems = sum(wins) + ties
       colors = ['#2E86AB', '#A23B72']
       
       bars = plt.bar(algorithms, wins, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
       
       # Customize plot
       plt.ylabel('Number of Problems Won', fontsize=14, fontweight='bold')
       plt.title(f'Algorithm Performance Comparison\n(n={total_problems} BBOB benchmark problems)', 
                fontsize=16, fontweight='bold', pad=20)
       
       # Add value labels on bars
       for bar, win_count in zip(bars, wins):
           height = bar.get_height()
           percentage = (win_count / total_problems * 100) if total_problems > 0 else 0
           plt.text(bar.get_x() + bar.get_width()/2, height + max(wins) * 0.02, 
                   f'{win_count}\n({percentage:.1f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
       
       # Add statistical info
       p_value = self.statistical_results.get('wilcoxon_test', {}).get('p_value', None)
       if p_value is not None and isinstance(p_value, (int, float)):
           p_text = f'Wilcoxon p-value: {p_value:.3f}'
       else:
           p_text = 'Wilcoxon p-value: N/A'
           
       max_win = max(wins) if wins and max(wins) > 0 else 1
       plt.text(0.5, 0.85, p_text, 
               ha='center', transform=plt.gca().transAxes, 
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
               fontsize=11)
       
       # Add ties information
       if ties > 0:
           tie_percentage = (ties / total_problems * 100) if total_problems > 0 else 0
           plt.text(0.5, 0.75, f'Ties: {ties} ({tie_percentage:.1f}%)', 
                   ha='center', transform=plt.gca().transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   fontsize=10)
       
       plt.ylim(0, max_win * 1.2)
       plt.grid(axis='y', alpha=0.3)
       plt.tight_layout()
       
       if save_plots:
           plot_path = Path(plot_dir)
           plot_path.mkdir(exist_ok=True)
           png_file = plot_path / 'win_loss_comparison.png'
           pdf_file = plot_path / 'win_loss_comparison.pdf'
           plt.savefig(png_file, dpi=300, bbox_inches='tight')
           plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
           print(f"   📈 Saved: {png_file}")
           print(f"   📈 Saved: {pdf_file}")
       plt.show()
   
   def _plot_rank_differences(self, wilcoxon: Dict, save_plots: bool, plot_dir: str):
       """Create rank difference distribution histogram."""
       plt.figure(figsize=(12, 8))
       
       rank_diffs = wilcoxon.get('rank_differences', [])
       if not rank_diffs:
           print("Warning: No rank differences data available for plotting")
           return
       
       # Create histogram
       bins = [-1.5, -0.5, 0.5, 1.5]
       counts, _, patches = plt.hist(rank_diffs, bins=bins, alpha=0.7, 
                                    color='skyblue', edgecolor='black', linewidth=1.5)
       
       # Color bars differently
       patches[0].set_facecolor('#2E86AB')  # TabPFN wins - blue
       if len(patches) > 1:
           patches[1].set_facecolor('#888888')  # Ties - gray
       if len(patches) > 2:
           patches[2].set_facecolor('#A23B72')  # Vanilla wins - red
       
       # Customize plot
       plt.xlabel('Rank Difference (Vanilla rank - TabPFN rank)', fontsize=14, fontweight='bold')
       plt.ylabel('Number of Problems', fontsize=14, fontweight='bold')
       plt.title('Distribution of Algorithm Ranking Differences\n(Negative = TabPFN wins, Positive = Vanilla wins)', 
                fontsize=16, fontweight='bold', pad=20)
       
       # Custom x-axis labels
       plt.xticks([-1, 0, 1], ['TabPFN\nWins\n(-1)', 'Perfect\nTie\n(0)', 'Vanilla\nWins\n(+1)'], fontsize=12)
       
       # Add count labels on bars
       x_positions = [-1, 0, 1]  # Positions for all 3 bins
       for i, count in enumerate(counts):
           if count > 0 and i < len(x_positions):
               x_pos = x_positions[i]
               plt.text(x_pos, count + 2, f'{int(count)}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=12)
       
       # Add statistics
       median_diff = wilcoxon.get('median_difference', 0)
       p_value = wilcoxon.get('p_value', 'N/A')
       
       # Add median line
       plt.axvline(median_diff, color='red', linestyle='--', linewidth=2, 
                  label=f'Median difference = {median_diff}')
       
       # Add statistics box
       stats_text = f'Wilcoxon Signed-Rank Test\np-value: {p_value:.3f}\nMedian difference: {median_diff}'
       plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
               verticalalignment='top', fontsize=11)
       
       plt.legend()
       plt.grid(axis='y', alpha=0.3)
       plt.tight_layout()
       
       if save_plots:
           plot_path = Path(plot_dir)
           plot_path.mkdir(exist_ok=True)
           png_file = plot_path / 'rank_difference_distribution.png'
           pdf_file = plot_path / 'rank_difference_distribution.pdf'
           plt.savefig(png_file, dpi=300, bbox_inches='tight')
           plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
           print(f"   📊 Saved: {png_file}")
           print(f"   📊 Saved: {pdf_file}")
       plt.show()
   
   def _plot_performance_by_dimension(self, save_plots: bool, plot_dir: str):
       """Create performance comparison by dimension."""
       # Extract win rates by dimension
       dimensions = [2, 5, 20, 40]
       tabpfn_wins_by_dim = []
       vanilla_wins_by_dim = []
       total_by_dim = []
       
       for dim in dimensions:
           tabpfn_wins = 0
           vanilla_wins = 0
           total = 0
           
           for problem_key, rankings in self.rankings.items():
               if f'_dim{dim}_' in problem_key:
                   total += 1
                   if rankings.get('tabpfn', 2) < rankings.get('vanilla', 1):
                       tabpfn_wins += 1
                   else:
                       vanilla_wins += 1
           
           tabpfn_wins_by_dim.append(tabpfn_wins / total * 100 if total > 0 else 0)
           vanilla_wins_by_dim.append(vanilla_wins / total * 100 if total > 0 else 0)
           total_by_dim.append(total)
       
       # Create the plot
       fig, ax = plt.subplots(figsize=(12, 8))
       
       x = np.arange(len(dimensions))
       width = 0.35
       
       bars1 = ax.bar(x - width/2, tabpfn_wins_by_dim, width, label='TabPFN BO', 
                     color='#2E86AB', alpha=0.8, edgecolor='black')
       bars2 = ax.bar(x + width/2, vanilla_wins_by_dim, width, label='Vanilla BO', 
                     color='#A23B72', alpha=0.8, edgecolor='black')
       
       # Customize plot
       ax.set_xlabel('Problem Dimension', fontsize=14, fontweight='bold')
       ax.set_ylabel('Win Rate (%)', fontsize=14, fontweight='bold')
       ax.set_title('Algorithm Performance by Problem Dimension', fontsize=16, fontweight='bold', pad=20)
       ax.set_xticks(x)
       ax.set_xticklabels(dimensions)
       ax.legend(fontsize=12)
       ax.grid(axis='y', alpha=0.3)
       
       # Add value labels on bars
       for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
           height1 = bar1.get_height()
           height2 = bar2.get_height()
           
           ax.text(bar1.get_x() + bar1.get_width()/2, height1 + 1, 
                  f'{height1:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
           ax.text(bar2.get_x() + bar2.get_width()/2, height2 + 1, 
                  f'{height2:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
           
           # Add sample size
           ax.text(i, -8, f'n={total_by_dim[i]}', ha='center', va='top', fontsize=10)
       
       ax.set_ylim(0, 100)
       plt.tight_layout()
       
       if save_plots:
           plot_path = Path(plot_dir)
           plot_path.mkdir(exist_ok=True)
           png_file = plot_path / 'performance_by_dimension.png'
           pdf_file = plot_path / 'performance_by_dimension.pdf'
           plt.savefig(png_file, dpi=300, bbox_inches='tight')
           plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
           print(f"   📈 Saved: {png_file}")
           print(f"   📈 Saved: {pdf_file}")
       plt.show()
   
   def run_complete_analysis(self, output_file: Optional[str] = None, create_plots: bool = True) -> Dict:
       """
       Run the complete statistical analysis pipeline.
       
       Args:
           output_file: Optional file to save results
           create_plots: Whether to create visualization plots
           
       Returns:
           Dictionary containing all analysis results
       """
       print("="*80)
       print("STATISTICAL ANALYSIS OF BAYESIAN OPTIMIZATION BENCHMARK")
       print("="*80)
       
       # Step 1: Collect all results
       print("\n1. Collecting benchmark results...")
       self.collect_all_results()
       
       if not self.performance_data:
           print("Error: No data collected. Check your results directory.")
           return {}
       
       # Step 2: Calculate rankings
       print("\n2. Calculating statistical rankings...")
       self.calculate_rankings()
       
       # Step 3: Perform statistical tests
       print("\n3. Performing statistical significance tests...")
       friedman_results = self.perform_friedman_test()
       wilcoxon_results = self.perform_wilcoxon_test()
       
       # Step 4: Calculate summary statistics
       print("\n4. Calculating summary statistics...")
       summary_stats = self.calculate_summary_statistics()
       
       # Compile all results
       complete_results = {
           'analysis_timestamp': datetime.now().isoformat(),
           'summary_statistics': summary_stats,
           'friedman_test': friedman_results,
           'wilcoxon_test': wilcoxon_results,
           'problem_rankings': self.rankings,
           'raw_performance_data_keys': list(self.performance_data.keys())
       }
       
       # Save results if requested
       if output_file:
           output_path = Path(output_file)
           output_path.parent.mkdir(parents=True, exist_ok=True)
           with open(output_path, 'w') as f:
               json.dump(complete_results, f, indent=2)
           print(f"\n5. Results saved to: {output_path}")
       
       self.statistical_results = complete_results
       
       # Step 6: Create visualizations
       if create_plots:
           print("\n6. Creating thesis visualizations...")
           self.create_visualizations()
       
       return complete_results
   
   def print_analysis_report(self):
       """Print a comprehensive analysis report to console."""
       if not self.statistical_results:
           print("Error: No analysis results available. Run run_complete_analysis() first.")
           return
       
       results = self.statistical_results
       summary = results.get('summary_statistics', {})
       friedman = results.get('friedman_test', {})
       wilcoxon = results.get('wilcoxon_test', {})
       
       print("\n" + "="*80)
       print("📊 COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
       print("="*80)
       
       # Dataset summary
       print(f"\n📋 DATASET SUMMARY")
       print("-" * 40)
       print(f"Total problem configurations: {summary.get('n_problems', 'Unknown')}")
       print(f"Total optimization runs: {summary.get('n_total_runs', 'Unknown')}")
       
# Algorithm ranking summary
       avg_ranks = summary.get('average_ranks', {})
       rank_std = summary.get('rank_std', {})
       
       print(f"\n🏆 ALGORITHM RANKING SUMMARY")
       print("-" * 40)
       if avg_ranks.get('vanilla') and avg_ranks.get('tabpfn'):
           print(f"{'Algorithm':<15} {'Avg Rank':<12} {'Std Dev':<12} {'Performance':<15}")
           print("-" * 60)
           
           vanilla_rank = avg_ranks['vanilla']
           tabpfn_rank = avg_ranks['tabpfn']
           vanilla_std = rank_std.get('vanilla', 0)
           tabpfn_std = rank_std.get('tabpfn', 0)
           
           # Determine best algorithm
           best_algo = "👑 BEST" if vanilla_rank < tabpfn_rank else ""
           worst_algo = "👑 BEST" if tabpfn_rank < vanilla_rank else ""
           
           print(f"{'Vanilla BO':<15} {vanilla_rank:<12.3f} {vanilla_std:<12.3f} {best_algo:<15}")
           print(f"{'TabPFN BO':<15} {tabpfn_rank:<12.3f} {tabpfn_std:<12.3f} {worst_algo:<15}")
       
       # Win/Loss statistics
       wlt = summary.get('win_loss_tie', {})
       if wlt:
           print(f"\n🥊 WIN/LOSS/TIE ANALYSIS")
           print("-" * 40)
           print(f"TabPFN wins:    {wlt.get('tabpfn_wins', 0):3d} ({wlt.get('tabpfn_win_rate', 0)*100:5.1f}%)")
           print(f"Vanilla wins:   {wlt.get('vanilla_wins', 0):3d} ({wlt.get('vanilla_win_rate', 0)*100:5.1f}%)")
           print(f"Ties:           {wlt.get('ties', 0):3d}")
       
       # Statistical significance tests
       print(f"\n📈 STATISTICAL SIGNIFICANCE TESTS")
       print("-" * 40)
       
       # Friedman test results
       if 'error' not in friedman:
           significance_stars = "***" if friedman.get('p_value', 1) < 0.001 else \
                              "**" if friedman.get('p_value', 1) < 0.01 else \
                              "*" if friedman.get('p_value', 1) < 0.05 else ""
           
           print(f"Friedman Test:")
           print(f"  Statistic: χ² = {friedman.get('statistic', 'N/A'):.4f}")
           print(f"  p-value:   p = {friedman.get('p_value', 'N/A'):.6f} {significance_stars}")
           print(f"  Result:    {friedman.get('interpretation', 'N/A')}")
       else:
           print(f"Friedman Test: {friedman.get('error', 'Failed')}")
           if friedman.get('note'):
               print(f"  Note: {friedman.get('note')}")
       
       print()
       
       # Wilcoxon test results
       if 'error' not in wilcoxon:
           significance_stars = "***" if wilcoxon.get('p_value', 1) < 0.001 else \
                              "**" if wilcoxon.get('p_value', 1) < 0.01 else \
                              "*" if wilcoxon.get('p_value', 1) < 0.05 else ""
           
           print(f"Wilcoxon Signed-Rank Test:")
           print(f"  Statistic: W = {wilcoxon.get('statistic', 'N/A'):.4f}")
           print(f"  p-value:   p = {wilcoxon.get('p_value', 'N/A'):.6f} {significance_stars}")
           print(f"  Effect size: Median rank difference = {wilcoxon.get('median_difference', 'N/A'):.3f}")
           if wilcoxon.get('n_ties', 0) > 0:
               print(f"  Ties: {wilcoxon.get('n_ties')} out of {wilcoxon.get('n_problems')} problems")
           print(f"  Result:    {wilcoxon.get('interpretation', 'N/A')}")
       else:
           print(f"Wilcoxon Test: {wilcoxon.get('error', 'Failed')}")
       
       # Performance statistics
       perf_stats = summary.get('performance_stats', {})
       if perf_stats:
           print(f"\n📊 RAW PERFORMANCE STATISTICS")
           print("-" * 40)
           print(f"{'Algorithm':<15} {'Mean Final':<15} {'Std Dev':<15}")
           print("-" * 50)
           
           vanilla_mean = perf_stats.get('vanilla_mean_final', 'N/A')
           tabpfn_mean = perf_stats.get('tabpfn_mean_final', 'N/A')
           vanilla_std = perf_stats.get('vanilla_std_final', 'N/A')
           tabpfn_std = perf_stats.get('tabpfn_std_final', 'N/A')
           
           if isinstance(vanilla_mean, (int, float)):
               print(f"{'Vanilla BO':<15} {vanilla_mean:<15.6e} {vanilla_std:<15.6e}")
           else:
               print(f"{'Vanilla BO':<15} {vanilla_mean:<15} {vanilla_std:<15}")
               
           if isinstance(tabpfn_mean, (int, float)):
               print(f"{'TabPFN BO':<15} {tabpfn_mean:<15.6e} {tabpfn_std:<15.6e}")
           else:
               print(f"{'TabPFN BO':<15} {tabpfn_mean:<15} {tabpfn_std:<15}")
       
       # Conclusion
       print(f"\n🎯 CONCLUSIONS")
       print("-" * 40)
       
       if wilcoxon.get('significant', False):
           if wilcoxon.get('median_difference', 0) > 0:
               print("✅ TabPFN BO significantly outperforms Vanilla BO")
               print(f"   (p = {wilcoxon.get('p_value', 'N/A'):.6f}, median rank difference = {wilcoxon.get('median_difference', 'N/A'):.3f})")
           else:
               print("✅ Vanilla BO significantly outperforms TabPFN BO")
               print(f"   (p = {wilcoxon.get('p_value', 'N/A'):.6f}, median rank difference = {wilcoxon.get('median_difference', 'N/A'):.3f})")
       else:
           print("❌ No statistically significant difference found between algorithms")
           if wilcoxon.get('p_value'):
               print(f"   (p = {wilcoxon.get('p_value', 'N/A'):.6f})")
           
           # Additional context for small datasets
           if wilcoxon.get('n_problems', 0) < 6:
               print(f"   ⚠️  Small sample size (n={wilcoxon.get('n_problems', 0)}) may limit statistical power")
       
       print("\n" + "="*80)


def main():
   """
   Example usage of the StatisticalAnalyzer with plotting.
   """
   # Path to your benchmark results
   results_path = "CONVPLOTS"  # Update this path
   
   # Create analyzer
   analyzer = StatisticalAnalyzer("C:/Users/semap/Documents/Uni/Theesis/CONVPLOTS")
   
   # Run complete analysis with plotting enabled
   print("🚀 Starting comprehensive statistical analysis...")
   results = analyzer.run_complete_analysis(
       output_file="statistical_analysis_results.json",
       create_plots=True  # This will create and save all the mandatory plots
   )
   
   # Print comprehensive report
   analyzer.print_analysis_report()
   
   # Summary of outputs
   print(f"\n🎉 ANALYSIS COMPLETE!")
   print(f"📄 Statistical results saved to: statistical_analysis_results.json")
   print(f"📊 Visualization plots saved to: thesis_plots/ directory")
   print(f"   - win_loss_comparison.png/.pdf")
   print(f"   - rank_difference_distribution.png/.pdf")
   print(f"   - performance_by_dimension.png/.pdf")
   
   return results


if __name__ == "__main__":
   main()