import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

def process_dat_file(file_path):
    """
    Process a .dat file and return data in the format expected by plot_convergence.
    """
    df = pd.read_csv(file_path, delimiter=' ', skipinitialspace=True)
    
    # Create the data structure for plotting
    run_data = {
        'evals': df['evaluations'].tolist(),
        'best_so_far': df['raw_y_best'].tolist(),  # Already regret values
        'final_value': df['raw_y_best'].iloc[-1]
    }
    
    return run_data

def plot_function_comparison(tabpfn_data, tabpfn_variance_data, problem_id, dimension, save_path):
    """
    Plot comparison between TabPFN and TabPFN_Variance for a specific function and dimension.
    Shows mean convergence curves with standard error bands.
    """
    plt.clf()
    plt.close('all')
    
    # Set up matplotlib for high-quality plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 6
    })
    
    # Create new figure with larger size for better readability
    fig = plt.figure(figsize=(14, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Define custom colors and distinct line styles
    algorithm_styles = {
        'TabPFN BO': {'color': '#015ee8', 'linestyle': '-'},      # Custom blue
        'TabPFN BO Variance': {'color': '#e80101', 'linestyle': '--'}       # Custom red
    }
    
    # Prepare algorithm data
    algorithm_data = {}
    if tabpfn_data:
        algorithm_data["TabPFN BO"] = tabpfn_data
    if tabpfn_variance_data:
        algorithm_data["TabPFN BO Variance"] = tabpfn_variance_data
    
    if not algorithm_data:
        print(f"Warning: No data available for plotting function {problem_id}")
        plt.close(fig)
        return None
    
    # Plot each algorithm with standard error
    for algorithm_name, algo_data in algorithm_data.items():
        if algorithm_name not in algorithm_styles:
            print(f"Warning: No style defined for {algorithm_name}, skipping")
            continue
            
        # Calculate max length for this algorithm's data
        max_length = max(len(data['best_so_far']) for data in algo_data)
        regret_matrix = []
        
        # Process each run for this algorithm
        for data in algo_data:
            # Use raw regret values directly
            regret_values = data['best_so_far']
            
            # Pad shorter runs with the last value
            if len(regret_values) < max_length:
                last_value = regret_values[-1] if regret_values else 1e-12
                regret_values.extend([last_value] * (max_length - len(regret_values)))
            
            regret_matrix.append(regret_values)
        
        regret_matrix = np.array(regret_matrix)
        
        if len(regret_matrix) == 0:
            print(f"Warning: No valid data for {algorithm_name}")
            continue
        
        # Calculate mean and standard error
        mean_regret = np.mean(regret_matrix, axis=0)
        std_regret = np.std(regret_matrix, axis=0)
        n_runs = len(algo_data)
        se_regret = std_regret / np.sqrt(n_runs)  # Standard error
        
        x_vals = np.array(range(1, max_length + 1))
        style = algorithm_styles[algorithm_name]
        
        # Add standard error bands first (so they appear behind the line)
        ax.fill_between(
            x_vals,
            mean_regret - se_regret,
            mean_regret + se_regret,
            color=style['color'],
            alpha=0.25,
            zorder=1
        )
        
        # Plot mean curve
        ax.plot(
            x_vals, 
            mean_regret, 
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=1.5,
            label=f'{algorithm_name} (n={n_runs})',
            zorder=2
        )
    
    # Calculate reasonable y-axis limits from regret data
    all_regret_values = []
    for algo_data in algorithm_data.values():
        for data in algo_data:
            all_regret_values.extend(data['best_so_far'])
    
    if not all_regret_values:
        print(f"Warning: No regret values for plotting function {problem_id}")
        plt.close(fig)
        return None
    
    # Handle y-axis limits for regret (all values should be positive for log scale)
    y_min = min(all_regret_values)
    y_max = max(all_regret_values)
    
    # Ensure minimum regret is positive for log scale
    if y_min <= 0:
        y_min = 1e-12
    
    # Handle case where y_min == y_max
    if y_min == y_max:
        y_min = y_min * 0.1 if y_min > 0 else 1e-12
        y_max = y_max * 10.0 if y_max > 0 else 1.0
    
    # Add padding in log space
    try:
        log_range = np.log10(y_max) - np.log10(y_min)
        if np.isfinite(log_range) and log_range > 0:
            y_min = 10 ** (np.log10(y_min) - 0.1 * log_range)
            y_max = 10 ** (np.log10(y_max) + 0.1 * log_range)
        else:
            y_min = y_min * 0.5
            y_max = y_max * 2.0
    except (ValueError, ZeroDivisionError):
        y_min = y_min * 0.5
        y_max = y_max * 2.0
    
    # Final safety check for valid limits
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max):
        print(f"Warning: Invalid axis limits for function {problem_id}, using default")
        y_min, y_max = ax.get_ylim()
    
    # Set axis limits
    try:
        ax.set_ylim(y_min, y_max)
    except ValueError as e:
        print(f"Warning: Could not set axis limits for function {problem_id}: {e}")
    
    # Improve y-axis readability for log scale
    # Uses scientific notation (e.g., 1×10⁻⁶) instead of small exponential text
    # Limits number of ticks to prevent crowding and improves label size
    formatter = ticker.LogFormatterMathtext(base=10, labelOnlyBase=False)
    ax.yaxis.set_major_formatter(formatter)
    
    # Control number of major ticks to avoid clutter (max 6-8 ticks)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
    
    # Increase tick label font size for better readability
    ax.tick_params(axis='y', which='major', labelsize=13, width=1.5, length=6)
    ax.tick_params(axis='x', which='major', labelsize=13, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', length=3, width=1)
    
    # Force x-axis to show only integer evaluation numbers with specific step sizes
    if dimension <= 5:
        step_size = 5  # Steps of 5 for lower dimensions
    else:
        step_size = 25  # Steps of 25 for higher dimensions
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step_size))
    ax.xaxis.set_minor_locator(ticker.NullLocator())  # Remove minor ticks on x-axis
    
    # Customize plot with regret-based labels and improved styling
    ax.set_title(f"Convergence Analysis: BBOB Function {problem_id} ({dimension}D)", 
                fontweight='bold', pad=20)
    
    # Improved grid styling
    ax.grid(True, which='major', ls='-', alpha=0.3, color='gray')
    ax.grid(True, which='minor', ls=':', alpha=0.2, color='gray')
    
    # Add subtle background color and improve layout
    ax.set_facecolor('#fafafa')
    
    # Adjust layout with more padding
    fig.tight_layout(pad=3.0)
    
    # Save plot with high quality
    fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', 
                pad_inches=0.3, facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Print summary statistics for regret across algorithms
    print("\nComparison of final regret across algorithms:")
    for algorithm_name, algo_data in algorithm_data.items():
        final_regrets = [data['best_so_far'][-1] for data in algo_data]
        print(f"{algorithm_name:20}: {np.mean(final_regrets):.6e} ± {np.std(final_regrets):.6e}")
    
    print(f"    📈 Regret-based comparison plot saved: {save_path.name}")
    
    return fig

def main(dimension, problem_id):
    """
    Main function to process and plot data for a specific dimension and function ID.
    
    Args:
        dimension (int): Problem dimension
        problem_id (int): BBOB function ID (1-24)
    """
    # Base directory for data
    base_dir = Path("bbob_benchmark_results")
    dim_dir = base_dir / f"dim_{dimension}"
    
    print(f"Looking for data in: {dim_dir}")
    
    if not dim_dir.exists():
        print(f"Error: Directory {dim_dir} does not exist!")
        return
    
    # Process .dat files from both algorithms
    tabpfn_data = []
    tabpfn_variance_data = []
    
    # Read TabPFN BO .dat files
    tabpfn_dir = dim_dir / "tabpfn"
    print(f"Searching tabpfn directory: {tabpfn_dir}")
    if tabpfn_dir.exists():
        # Look for .dat files in nested structure: tabpfn/{run_folder}/data_f{X}_{FunctionName}/IOHprofiler_f{X}_DIM{Y}.dat
        for run_folder in tabpfn_dir.iterdir():
            if run_folder.is_dir() and f"f{problem_id}_dim{dimension}" in run_folder.name:
                print(f"  Found matching tabpfn run folder: {run_folder.name}")
                # Look for data folder inside run folder
                data_folders = list(run_folder.glob(f"data_f{problem_id}_*"))
                for data_folder in data_folders:
                    if data_folder.is_dir():
                        dat_file = data_folder / f"IOHprofiler_f{problem_id}_DIM{dimension}.dat"
                        if dat_file.exists():
                            try:
                                run_data = process_dat_file(dat_file)
                                tabpfn_data.append(run_data)
                                print(f"    ✓ Found tabpfn data file: {dat_file}")
                            except Exception as e:
                                print(f"    ✗ Error processing tabpfn .dat file {dat_file}: {e}")
                        else:
                            print(f"    ✗ Expected dat file not found: {dat_file}")
    else:
        print(f"  TabPFN directory does not exist: {tabpfn_dir}")
    
    # Read TabPFN BO Variance .dat files  
    tabpfn_variance_dir = dim_dir / "tabpfn_variance"
    print(f"Searching tabpfn_variance directory: {tabpfn_variance_dir}")
    if tabpfn_variance_dir.exists():
        # Look for .dat files in nested structure: tabpfn_variance/{run_folder}/data_f{X}_{FunctionName}/IOHprofiler_f{X}_DIM{Y}.dat
        for run_folder in tabpfn_variance_dir.iterdir():
            if run_folder.is_dir() and f"f{problem_id}_dim{dimension}" in run_folder.name:
                print(f"  Found matching tabpfn_variance run folder: {run_folder.name}")
                # Look for data folder inside run folder
                data_folders = list(run_folder.glob(f"data_f{problem_id}_*"))
                for data_folder in data_folders:
                    if data_folder.is_dir():
                        dat_file = data_folder / f"IOHprofiler_f{problem_id}_DIM{dimension}.dat"
                        if dat_file.exists():
                            try:
                                run_data = process_dat_file(dat_file)
                                tabpfn_variance_data.append(run_data)
                                print(f"    ✓ Found tabpfn_variance data file: {dat_file}")
                            except Exception as e:
                                print(f"    ✗ Error processing tabpfn_variance .dat file {dat_file}: {e}")
                        else:
                            print(f"    ✗ Expected dat file not found: {dat_file}")
    else:
        print(f"  TabPFN Variance directory does not exist: {tabpfn_variance_dir}")
    
    print(f"Data summary: TabPFN={len(tabpfn_data)} runs, TabPFN Variance={len(tabpfn_variance_data)} runs")
    
    if tabpfn_data or tabpfn_variance_data:
        plot_path = dim_dir / f"D{dimension} ({problem_id}).png"
        plot_function_comparison(
            tabpfn_data=tabpfn_data,
            tabpfn_variance_data=tabpfn_variance_data,
            problem_id=problem_id,
            dimension=dimension,
            save_path=plot_path
        )
    else:
        print(f"No data found for function {problem_id} in dimension {dimension}")

if __name__ == "__main__":
    # ========================================
    # PARAMETERS TO MODIFY
    # ========================================
    DIMENSIONS = [2]      # Problem dimensions (matching ConvergenceSmoothened.py)
    FUNCTION_IDS = range(1, 25)  # List of BBOB function IDs (1-24) to process
    # ========================================
    
    # Validate function IDs
    invalid_ids = [fid for fid in FUNCTION_IDS if not 1 <= fid <= 24]
    if invalid_ids:
        print(f"Error: Invalid function IDs {invalid_ids}. All function IDs must be between 1 and 24")
        exit(1)
    
    # Validate dimensions
    if not DIMENSIONS:
        print("Error: No dimensions specified")
        exit(1)
    
    total_tasks = len(DIMENSIONS) * len(FUNCTION_IDS)
    print(f"Processing {len(FUNCTION_IDS)} functions across {len(DIMENSIONS)} dimensions...")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Function IDs: {list(FUNCTION_IDS)}")
    print(f"Total tasks: {total_tasks}")
    print("=" * 50)
    
    # Process each dimension and function combination
    task_count = 0
    for dimension in DIMENSIONS:
        print(f"\n🔍 PROCESSING DIMENSION {dimension}")
        print("-" * 30)
        
        for i, problem_id in enumerate(FUNCTION_IDS, 1):
            task_count += 1
            print(f"\n[{task_count}/{total_tasks}] Dimension {dimension}, Function {problem_id}...")
            main(dimension, problem_id)
    
    print(f"\n✅ Completed processing all {total_tasks} tasks!")
    print(f"📁 Results saved in bbob_benchmark_results/dim_X/ directories") 