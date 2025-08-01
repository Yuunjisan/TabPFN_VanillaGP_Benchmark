"""
BBOB Function Group Analysis with Fischer's Exact Test
Tests whether TabPFN-BO performs significantly better than GP-BO on different BBOB function categories.
Specifically tests the hypothesis that TabPFN performs better on multimodal vs unimodal functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import fisher_exact
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class BBOBGroupAnalyzer:
    """
    Analyzes TabPFN vs Vanilla BO performance by BBOB function groups using Fischer's exact test.
    """
    
    # BBOB function groups according to standard categorization
    BBOB_GROUPS = {
        1: {
            'name': 'Separable Functions',
            'functions': [1, 2, 3, 4, 5],
            'type': 'unimodal',
            'description': 'Separable functions'
        },
        2: {
            'name': 'Low/Moderate Conditioning',
            'functions': [6, 7, 8, 9],
            'type': 'unimodal', 
            'description': 'Functions with low or moderate conditioning'
        },
        3: {
            'name': 'High Conditioning Unimodal',
            'functions': [10, 11, 12, 13, 14],
            'type': 'unimodal',
            'description': 'Functions with high conditioning and unimodal'
        },
        4: {
            'name': 'Multi-modal Adequate Structure',
            'functions': [15, 16, 17, 18, 19],
            'type': 'multimodal',
            'description': 'Multi-modal functions with adequate global structure'
        },
        5: {
            'name': 'Multi-modal Weak Structure', 
            'functions': [20, 21, 22, 23, 24],
            'type': 'multimodal',
            'description': 'Multi-modal functions with weak global structure'
        }
    }
    
    def __init__(self, results_dir: str = "FinalPlots", significance_threshold: float = 0.1):
        """
        Initialize the BBOB group analyzer.
        
        Args:
            results_dir: Path to the benchmark results directory
            significance_threshold: Threshold for determining "significantly better" performance
        """
        self.results_dir = Path(results_dir)
        self.significance_threshold = significance_threshold  # 10% relative improvement threshold
        self.performance_data = {}
        self.group_results = {}
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
        Load and process a single .dat file (same as Statistical_Tests.py).
        """
        try:
            # Read the .dat file
            df = pd.read_csv(file_path, delimiter=' ', skipinitialspace=True)
            
            # Extract metadata from file path
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
                    parent_dir = file_path.parent.parent.name
                    if '_inst' in parent_dir and '_rep' in parent_dir:
                        inst_part = parent_dir.split('_inst')[1].split('_rep')[0]
                        rep_part = parent_dir.split('_rep')[1]
                        instance = int(inst_part)
                        repetition = int(rep_part)
                    else:
                        instance = 1
                        repetition = 1
                        
                except (ValueError, IndexError):
                    return None
            else:
                return None
            
            # Process the data
            if 'raw_y_best' not in df.columns:
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
            
        except Exception:
            return None
    
    def collect_all_results(self) -> Dict:
        """
        Collect all .dat files and organize by problem configuration.
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
        
        # Filter out problems that don't have both algorithms
        filtered_data = {}
        for problem_key, algo_data in organized_data.items():
            if len(algo_data['vanilla']) > 0 and len(algo_data['tabpfn']) > 0:
                filtered_data[problem_key] = algo_data
            else:
                print(f"Warning: Skipping {problem_key} - missing data for one algorithm")
        
        print(f"Final dataset: {len(filtered_data)} problem configurations with both algorithms")
        self.performance_data = filtered_data
        return filtered_data
    
    def determine_significantly_better(self, tabpfn_values: List[float], vanilla_values: List[float]) -> bool:
        """
        Determine if TabPFN performs "significantly better" than Vanilla BO.
        Uses relative improvement threshold.
        
        Args:
            tabpfn_values: List of final best values for TabPFN
            vanilla_values: List of final best values for Vanilla BO
            
        Returns:
            True if TabPFN is significantly better
        """
        # Calculate medians (robust to outliers)
        tabpfn_median = np.median(tabpfn_values)
        vanilla_median = np.median(vanilla_values)
        
        # For minimization problems, lower is better
        # Calculate relative improvement: (vanilla - tabpfn) / |vanilla|
        if abs(vanilla_median) > 1e-12:  # Avoid division by zero
            relative_improvement = (vanilla_median - tabpfn_median) / abs(vanilla_median)
        else:
            # If vanilla_median is near zero, use absolute difference
            relative_improvement = vanilla_median - tabpfn_median
        
        # TabPFN is significantly better if it improves by more than threshold
        return relative_improvement > self.significance_threshold
    
    def analyze_by_groups(self) -> Dict:
        """
        Analyze performance by BBOB function groups.
        
        Returns:
            Dictionary containing group analysis results
        """
        if not self.performance_data:
            print("Error: No performance data loaded. Run collect_all_results() first.")
            return {}
        
        print("\nAnalyzing performance by BBOB function groups...")
        
        group_results = {}
        
        for group_id, group_info in self.BBOB_GROUPS.items():
            group_name = group_info['name']
            group_functions = group_info['functions']
            group_type = group_info['type']
            
            print(f"\nProcessing Group {group_id}: {group_name}")
            print(f"Functions: {group_functions} ({group_type})")
            
            successes = 0  # TabPFN significantly better
            total_comparisons = 0
            function_details = {}
            
            for function_id in group_functions:
                function_successes = 0
                function_total = 0
                
                # Look for all problem configurations with this function
                for problem_key, algo_data in self.performance_data.items():
                    if problem_key.startswith(f"f{function_id}_"):
                        # Extract performance values
                        tabpfn_values = [run['final_best'] for run in algo_data['tabpfn']]
                        vanilla_values = [run['final_best'] for run in algo_data['vanilla']]
                        
                        # Check if TabPFN is significantly better
                        is_significantly_better = self.determine_significantly_better(tabpfn_values, vanilla_values)
                        
                        if is_significantly_better:
                            function_successes += 1
                            successes += 1
                        
                        function_total += 1
                        total_comparisons += 1
                
                function_details[function_id] = {
                    'successes': int(function_successes),
                    'total': int(function_total),
                    'success_rate': float(function_successes / function_total if function_total > 0 else 0)
                }
                
                print(f"  f{function_id}: {function_successes}/{function_total} = {function_successes/function_total*100:.1f}%" if function_total > 0 else f"  f{function_id}: No data")
            
            # Calculate group statistics
            success_rate = successes / total_comparisons if total_comparisons > 0 else 0
            
            group_results[group_id] = {
                'group_info': group_info,
                'successes': int(successes),
                'total_comparisons': int(total_comparisons),
                'success_rate': float(success_rate),
                'function_details': function_details
            }
            
            print(f"Group {group_id} overall: {successes}/{total_comparisons} = {success_rate*100:.1f}%")
        
        self.group_results = group_results
        return group_results
    
    def perform_fisher_exact_tests(self) -> Dict:
        """
        Perform Fisher's exact test for each group to test if success rate > 50%.
        
        Returns:
            Dictionary containing Fisher's exact test results
        """
        if not self.group_results:
            print("Error: No group results available. Run analyze_by_groups() first.")
            return {}
        
        print("\nPerforming Fisher's exact tests...")
        
        fisher_results = {}
        
        for group_id, group_data in self.group_results.items():
            group_name = group_data['group_info']['name']
            successes = group_data['successes']
            total = group_data['total_comparisons']
            
            if total == 0:
                print(f"Group {group_id}: No data available")
                continue
            
            # Fisher's exact test: H0: success_rate = 0.5, H1: success_rate > 0.5
            # We use a 2x2 contingency table:
            # [successes, failures] vs [expected_successes, expected_failures]
            failures = total - successes
            expected_successes = total // 2
            expected_failures = total - expected_successes
            
            # Create contingency table
            contingency_table = np.array([
                [successes, failures],
                [expected_successes, expected_failures]
            ])
            
            # Perform one-tailed Fisher's exact test (alternative='greater')
            try:
                odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
                
                # Interpret results
                significant = p_value < 0.05
                interpretation = f"TabPFN significantly outperforms on {group_name}" if significant else f"No significant advantage for TabPFN on {group_name}"
                
                fisher_results[group_id] = {
                    'group_name': group_name,
                    'group_type': group_data['group_info']['type'],
                    'successes': int(successes),
                    'total': int(total),
                    'success_rate': float(group_data['success_rate']),
                    'odds_ratio': float(odds_ratio),
                    'p_value': float(p_value),
                    'significant': bool(significant),
                    'contingency_table': contingency_table.tolist(),
                    'interpretation': interpretation
                }
                
                print(f"Group {group_id} ({group_name}): p = {p_value:.4f}, OR = {odds_ratio:.4f} {'***' if significant else ''}")
                
            except Exception as e:
                print(f"Error in Fisher's exact test for group {group_id}: {e}")
                fisher_results[group_id] = {'error': str(e)}
        
        return fisher_results
    
    def test_unimodal_vs_multimodal(self) -> Dict:
        """
        Test the main hypothesis: TabPFN performs better on multimodal vs unimodal functions.
        
        Returns:
            Dictionary containing comparison results
        """
        if not self.group_results:
            print("Error: No group results available. Run analyze_by_groups() first.")
            return {}
        
        print("\nTesting unimodal vs multimodal hypothesis...")
        
        # Aggregate results by function type
        unimodal_successes = 0
        unimodal_total = 0
        multimodal_successes = 0
        multimodal_total = 0
        
        for group_id, group_data in self.group_results.items():
            group_type = group_data['group_info']['type']
            successes = group_data['successes']
            total = group_data['total_comparisons']
            
            if group_type == 'unimodal':
                unimodal_successes += successes
                unimodal_total += total
            elif group_type == 'multimodal':
                multimodal_successes += successes
                multimodal_total += total
        
        # Calculate success rates
        unimodal_rate = unimodal_successes / unimodal_total if unimodal_total > 0 else 0
        multimodal_rate = multimodal_successes / multimodal_total if multimodal_total > 0 else 0
        
        print(f"Unimodal functions (Groups 1-3): {unimodal_successes}/{unimodal_total} = {unimodal_rate*100:.1f}%")
        print(f"Multimodal functions (Groups 4-5): {multimodal_successes}/{multimodal_total} = {multimodal_rate*100:.1f}%")
        
        # Fisher's exact test comparing unimodal vs multimodal
        if unimodal_total > 0 and multimodal_total > 0:
            contingency_table = np.array([
                [multimodal_successes, multimodal_total - multimodal_successes],
                [unimodal_successes, unimodal_total - unimodal_successes]
            ])
            
            try:
                odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
                
                significant = p_value < 0.05
                interpretation = "TabPFN performs significantly better on multimodal functions" if significant else "No significant difference between unimodal and multimodal functions"
                
                comparison_results = {
                    'unimodal': {
                        'successes': int(unimodal_successes),
                        'total': int(unimodal_total),
                        'success_rate': float(unimodal_rate)
                    },
                    'multimodal': {
                        'successes': int(multimodal_successes),
                        'total': int(multimodal_total),
                        'success_rate': float(multimodal_rate)
                    },
                    'fisher_test': {
                        'odds_ratio': float(odds_ratio),
                        'p_value': float(p_value),
                        'significant': bool(significant),
                        'contingency_table': contingency_table.tolist(),
                        'interpretation': interpretation
                    }
                }
                
                print(f"Fisher's exact test (multimodal > unimodal): p = {p_value:.4f}, OR = {odds_ratio:.4f}")
                print(f"Result: {interpretation}")
                
                return comparison_results
                
            except Exception as e:
                print(f"Error in unimodal vs multimodal test: {e}")
                return {'error': str(e)}
        else:
            return {'error': 'Insufficient data for comparison'}
    
    def create_visualizations(self, save_plots: bool = True, plot_dir: str = "bbob_group_plots"):
        """
        Create visualizations for BBOB group analysis.
        """
        if not self.statistical_results:
            print("Error: No analysis results available. Run run_complete_analysis() first.")
            return
        
        # Create plot directory
        plot_path = Path(plot_dir)
        if save_plots:
            plot_path.mkdir(exist_ok=True)
            print(f"\n📁 Creating BBOB group visualizations in: {plot_path.absolute()}")
        
        # 1. Success rates by group
        self._plot_group_success_rates(save_plots, plot_dir)
        
        # 2. Unimodal vs Multimodal comparison
        self._plot_unimodal_vs_multimodal(save_plots, plot_dir)
        
        if save_plots:
            print(f"\n✅ All visualizations created and saved to: {plot_path.absolute()}")
    
    def _plot_group_success_rates(self, save_plots: bool, plot_dir: str):
        """Create success rate bar chart by BBOB group."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        fisher_results = self.statistical_results.get('fisher_tests', {})
        
        groups = []
        success_rates = []
        p_values = []
        colors = []
        
        for group_id in sorted(fisher_results.keys()):
            if isinstance(group_id, int):
                result = fisher_results[group_id]
                if 'error' not in result:
                    groups.append(f"Group {group_id}\n{result['group_name']}")
                    success_rates.append(result['success_rate'] * 100)
                    p_values.append(result['p_value'])
                    
                    # Color by significance and function type
                    if result['significant']:
                        colors.append('#2E86AB' if result['group_type'] == 'multimodal' else '#A23B72')
                    else:
                        colors.append('#888888')
        
        bars = ax.bar(groups, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add 50% reference line
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Chance Performance (50%)')
        
        # Customize plot
        ax.set_ylabel('TabPFN Success Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title('TabPFN Performance by BBOB Function Groups\n(Proportion of problems where TabPFN significantly outperforms Vanilla BO)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels and significance stars
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            significance_stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                   f'{height:.1f}%\n{significance_stars}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Multimodal (Significant)'),
            Patch(facecolor='#A23B72', label='Unimodal (Significant)'),
            Patch(facecolor='#888888', label='Not Significant'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Chance Performance')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_ylim(0, max(success_rates) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_plots:
            plot_path = Path(plot_dir)
            plot_path.mkdir(exist_ok=True)
            plt.savefig(plot_path / 'bbob_group_success_rates.png', dpi=300, bbox_inches='tight')
            plt.savefig(plot_path / 'bbob_group_success_rates.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_unimodal_vs_multimodal(self, save_plots: bool, plot_dir: str):
        """Create comparison plot for unimodal vs multimodal functions."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        comparison = self.statistical_results.get('unimodal_vs_multimodal', {})
        
        if 'error' in comparison:
            print("Cannot create unimodal vs multimodal plot: insufficient data")
            return
        
        categories = ['Unimodal\n(Groups 1-3)', 'Multimodal\n(Groups 4-5)']
        success_rates = [
            comparison['unimodal']['success_rate'] * 100,
            comparison['multimodal']['success_rate'] * 100
        ]
        colors = ['#A23B72', '#2E86AB']
        
        bars = ax.bar(categories, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add 50% reference line
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Chance Performance (50%)')
        
        # Customize plot
        ax.set_ylabel('TabPFN Success Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title('TabPFN Performance: Unimodal vs Multimodal Functions\n(Testing hypothesis: TabPFN better on multimodal functions)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add Fisher's test results
        fisher_test = comparison.get('fisher_test', {})
        if fisher_test:
            p_value = fisher_test.get('p_value', 'N/A')
            significance = "***" if isinstance(p_value, (int, float)) and p_value < 0.001 else \
                          "**" if isinstance(p_value, (int, float)) and p_value < 0.01 else \
                          "*" if isinstance(p_value, (int, float)) and p_value < 0.05 else ""
            
            stats_text = f"Fisher's Exact Test\np-value: {p_value:.4f} {significance}\n{fisher_test.get('interpretation', '')}"
            ax.text(0.5, 0.85, stats_text, transform=ax.transAxes, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), fontsize=11)
        
        ax.set_ylim(0, max(success_rates) * 1.3)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plot_path = Path(plot_dir)
            plot_path.mkdir(exist_ok=True)
            plt.savefig(plot_path / 'unimodal_vs_multimodal.png', dpi=300, bbox_inches='tight')
            plt.savefig(plot_path / 'unimodal_vs_multimodal.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, output_file: Optional[str] = None, create_plots: bool = True) -> Dict:
        """
        Run the complete BBOB group analysis pipeline.
        """
        print("="*80)
        print("BBOB FUNCTION GROUP ANALYSIS WITH FISCHER'S EXACT TEST")
        print("="*80)
        
        # Step 1: Collect all results
        print("\n1. Collecting benchmark results...")
        self.collect_all_results()
        
        if not self.performance_data:
            print("Error: No data collected. Check your results directory.")
            return {}
        
        # Step 2: Analyze by groups
        print("\n2. Analyzing performance by BBOB function groups...")
        self.analyze_by_groups()
        
        # Step 3: Perform Fisher's exact tests
        print("\n3. Performing Fisher's exact tests for each group...")
        fisher_results = self.perform_fisher_exact_tests()
        
        # Step 4: Test unimodal vs multimodal hypothesis
        print("\n4. Testing unimodal vs multimodal hypothesis...")
        comparison_results = self.test_unimodal_vs_multimodal()
        
        # Compile all results
        complete_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'significance_threshold': float(self.significance_threshold),
            'bbob_groups': self.BBOB_GROUPS,
            'group_results': self.group_results,
            'fisher_tests': fisher_results,
            'unimodal_vs_multimodal': comparison_results,
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
            print("\n6. Creating BBOB group visualizations...")
            self.create_visualizations()
        
        return complete_results
    
    def print_analysis_report(self):
        """Print a comprehensive BBOB group analysis report."""
        if not self.statistical_results:
            print("Error: No analysis results available. Run run_complete_analysis() first.")
            return
        
        results = self.statistical_results
        fisher_tests = results.get('fisher_tests', {})
        comparison = results.get('unimodal_vs_multimodal', {})
        
        print("\n" + "="*80)
        print("📊 BBOB FUNCTION GROUP ANALYSIS REPORT")
        print("="*80)
        
        # Dataset summary
        print(f"\n📋 ANALYSIS PARAMETERS")
        print("-" * 40)
        print(f"Significance threshold: {self.significance_threshold*100:.1f}% relative improvement")
        print(f"Total problem configurations: {len(self.performance_data)}")
        
        # Group-by-group results
        print(f"\n🎯 RESULTS BY BBOB FUNCTION GROUP")
        print("-" * 60)
        print(f"{'Group':<8} {'Type':<12} {'Success Rate':<12} {'p-value':<10} {'Significant':<12}")
        print("-" * 60)
        
        for group_id in sorted(fisher_tests.keys()):
            if isinstance(group_id, int):
                result = fisher_tests[group_id]
                if 'error' not in result:
                    group_type = result['group_type']
                    success_rate = result['success_rate'] * 100
                    p_value = result['p_value']
                    significant = "Yes ***" if result['significant'] else "No"
                    
                    print(f"{group_id:<8} {group_type:<12} {success_rate:<12.1f}% {p_value:<10.4f} {significant:<12}")
        
        # Main hypothesis test
        print(f"\n🔬 MAIN HYPOTHESIS TEST: UNIMODAL vs MULTIMODAL")
        print("-" * 60)
        
        if 'error' not in comparison:
            unimodal = comparison['unimodal']
            multimodal = comparison['multimodal']
            fisher_test = comparison['fisher_test']
            
            print(f"Unimodal functions (Groups 1-3):   {unimodal['success_rate']*100:.1f}% ({unimodal['successes']}/{unimodal['total']})")
            print(f"Multimodal functions (Groups 4-5): {multimodal['success_rate']*100:.1f}% ({multimodal['successes']}/{multimodal['total']})")
            print(f"\nFisher's Exact Test (H₁: multimodal > unimodal):")
            print(f"  p-value: {fisher_test['p_value']:.4f}")
            print(f"  Odds ratio: {fisher_test['odds_ratio']:.3f}")
            print(f"  Result: {fisher_test['interpretation']}")
        else:
            print("Error in hypothesis test:", comparison.get('error', 'Unknown error'))
        
        # Conclusions
        print(f"\n🎯 CONCLUSIONS")
        print("-" * 40)
        
        # Count significant groups by type
        significant_unimodal = sum(1 for result in fisher_tests.values() 
                                 if isinstance(result, dict) and 'error' not in result and 
                                 result.get('significant', False) and result.get('group_type') == 'unimodal')
        significant_multimodal = sum(1 for result in fisher_tests.values() 
                                   if isinstance(result, dict) and 'error' not in result and 
                                   result.get('significant', False) and result.get('group_type') == 'multimodal')
        
        print(f"✅ Significant groups where TabPFN outperforms:")
        print(f"   - Unimodal: {significant_unimodal}/3 groups")
        print(f"   - Multimodal: {significant_multimodal}/2 groups")
        
        if 'error' not in comparison and comparison['fisher_test']['significant']:
            print(f"✅ Main hypothesis CONFIRMED: TabPFN performs significantly better on multimodal functions")
        else:
            print(f"❌ Main hypothesis NOT CONFIRMED: No significant difference between unimodal and multimodal performance")
        
        print("\n" + "="*80)


def main():
    """
    Run BBOB function group analysis with Fischer's exact test.
    """
    # Path to your benchmark results (same as Statistical_Tests.py)
    results_path = "FinalPlots"
    
    # Create analyzer with 10% relative improvement threshold
    analyzer = BBOBGroupAnalyzer(results_path, significance_threshold=0.1)
    
    # Run complete analysis
    print("🚀 Starting BBOB function group analysis...")
    results = analyzer.run_complete_analysis(
        output_file="bbob_group_analysis_results.json",
        create_plots=True
    )
    
    # Print comprehensive report
    analyzer.print_analysis_report()
    
    # Summary of outputs
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📄 Results saved to: bbob_group_analysis_results.json")
    print(f"📊 Plots saved to: bbob_group_plots/ directory")
    print(f"   - bbob_group_success_rates.png/.pdf")
    print(f"   - unimodal_vs_multimodal.png/.pdf")
    
    return results


if __name__ == "__main__":
    main() 