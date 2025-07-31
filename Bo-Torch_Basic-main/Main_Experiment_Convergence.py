r"""
This script runs a comprehensive publication-quality benchmark comparing Bayesian Optimization algorithms
on BBOB functions across multiple dimensions, instances, and repetitions.
"""

### -------------------------------------------------------------
### IMPORT LIBRARIES/REPOSITORIES
###---------------------------------------------------------------

# Algorithm import
from Algorithms import Vanilla_BO
from Algorithms.BayesianOptimization.TabPFN_BO.TabPFN_BO import TabPFN_BO

# Standard libraries
import os
from pathlib import Path
from numpy.linalg import norm
import numpy as np
import pandas as pd
import traceback
import torch
import json
from datetime import datetime
import time

### ---------------------------------------------------------------
### DEVICE CONFIGURATION
### ---------------------------------------------------------------

# Device configuration - set to "cpu", "cuda", or "auto" for automatic detection
DEVICE = "cuda"  # Change this to force a specific device

# Check for GPU availability and configure device
if DEVICE == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = DEVICE

# IOH Experimenter libraries
try:
    from ioh import get_problem
    import ioh.iohcpp.logger as logger_lib
    from ioh.iohcpp.logger import Analyzer
    from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT, ALWAYS
except ModuleNotFoundError as e:
    print(e.args)
except Exception as e:
    print(e.args)

### ---------------------------------------------------------------
### BENCHMARK CONFIGURATION
### ---------------------------------------------------------------

# BBOB functions to test (can modify this list)
BBOB_FUNCTIONS = [1] #list(range(19, 25))  # Functions 1-24
DIMENSIONS = [2]  # Test dimensions
INSTANCES = [1, 2, 3]  # Problem instances
N_REPETITIONS = 5  # Number of runs with different seeds
BASE_SEED = 42  # Base seed for reproducibility

# Algorithms to compare
ALGORITHMS = ["vanilla", "tabpfn"]
ACQUISITION_FUNCTION = "expected_improvement"

# Budget configuration (adaptive based on dimension)
def get_budget(dimension): 
    """Calculate budget based on dimension"""
    return 10 * dimension  # Budget = 10*D 

def get_n_doe(dimension):
    """Calculate number of initial design points based on dimension"""
    return 3 * dimension

### ---------------------------------------------------------------
### EXPERIMENT MANAGEMENT
### ---------------------------------------------------------------

class BenchmarkManager:
    """Manages the comprehensive benchmark experiment"""
    
    def __init__(self, base_output_dir="bbob_benchmark_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create dimension-based folder structure
        for dim in DIMENSIONS:
            dim_dir = self.base_output_dir / f"dim_{dim}"
            dim_dir.mkdir(exist_ok=True)
            
            # Create algorithm subdirectories
            for algorithm in ALGORITHMS:
                algo_dir = dim_dir / algorithm
                algo_dir.mkdir(exist_ok=True)
        
        # Initialize experiment log
        self.experiment_log = {
            "start_time": datetime.now().isoformat(),
            "configuration": {
                "bbob_functions": BBOB_FUNCTIONS,
                "dimensions": DIMENSIONS,
                "instances": INSTANCES,
                "n_repetitions": N_REPETITIONS,
                "algorithms": ALGORITHMS,
                "acquisition_function": ACQUISITION_FUNCTION
            },
            "results": {}
        }
        
def run_single_optimization(algorithm, problem_id, dimension, instance, repetition, 
                          output_dir, fit_mode="fit_with_cache", device="auto"):
    """
    Run a single optimization experiment.
    
    Args:
        algorithm: "vanilla" or "tabpfn"
        problem_id: BBOB function ID (1-24)
        dimension: Problem dimension
        instance: Problem instance (1-3)
        repetition: Repetition number (1-N_REPETITIONS)
        output_dir: Directory to save results
        fit_mode: TabPFN fit mode
        device: Device to use ("cpu", "cuda", or "auto" for automatic detection)
    
    Returns:
        dict: Results summary
    """
    # Calculate seed for reproducibility
    seed = BASE_SEED + repetition
    
    # Set up problem
    problem = get_problem(problem_id, instance=instance, dimension=dimension)
    
    # Calculate budget and DoE
    budget = get_budget(dimension)
    n_DoE = get_n_doe(dimension)
    
    # Create unique logger for this run
    run_name = f"{algorithm}_f{problem_id}_dim{dimension}_inst{instance}_rep{repetition}"
    logger_dir = output_dir / run_name
    
    # Set up logger
    triggers = [Each(1), ON_IMPROVEMENT]
    run_logger = Analyzer(
        triggers=triggers,
        root=str(logger_dir.parent),
        folder_name=logger_dir.name,
        algorithm_name=f"{algorithm.title()} BO",
        algorithm_info=f"Function {problem_id}, Dim {dimension}, Instance {instance}, Rep {repetition}",
        additional_properties=[logger_lib.property.RAWYBEST],
        store_positions=True
    )
    
    problem.attach_logger(run_logger)
    
    try:
        # Common parameters
        common_params = {
            'budget': budget,
            'n_DoE': n_DoE,
            'acquisition_function': ACQUISITION_FUNCTION,
            'random_seed': seed,
            'maximisation': False,
            'verbose': False,  # Set to False for cleaner output during benchmark
            'device': device
        }
        
        # Create optimizer based on algorithm
        if algorithm == "vanilla":
            optimizer = Vanilla_BO(**common_params)
        elif algorithm == "tabpfn":
            optimizer = TabPFN_BO(
                **common_params,
                n_estimators=32,  # Conservative for large benchmark
                fit_mode=fit_mode,
                temperature=0.9
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Watch the optimizer
        run_logger.watch(optimizer, "acquisition_function_name")
        
        # Run optimization
        optimizer(problem=problem)
        
        # Collect results
        result = {
            'algorithm': algorithm,
            'problem_id': problem_id,
            'dimension': dimension,
            'instance': instance,
            'repetition': repetition,
            'seed': seed,
            'budget': budget,
            'n_DoE': n_DoE,
            'final_best': float(optimizer.current_best),
            'final_regret': float(problem.state.current_best.y - problem.optimum.y),
            'n_evaluations': len(optimizer.f_evals),
            'success': True
        }
        
    except Exception as e:
        result = {
            'algorithm': algorithm,
            'problem_id': problem_id,
            'dimension': dimension,
            'instance': instance,
            'repetition': repetition,
            'seed': seed,
            'success': False,
            'error': str(e)
        }
    
    finally:
        run_logger.close()
    
    return result

def process_dat_file(file_path):
    """
    Process a .dat file and return data in the format expected by analyze_function_comparison.
    Same function as in OG_convergence.py.
    """
    df = pd.read_csv(file_path, delimiter=' ', skipinitialspace=True)
    
    # Extract run number from directory or filename
    run_number = 1  # Default run number for benchmark
    
    # Create the data structure for analysis
    run_data = {
        'run': run_number,
        'evals': df['evaluations'].tolist(),
        'best_so_far': df['raw_y_best'].tolist(),
        'final_value': df['raw_y_best'].iloc[-1]
    }
    
    return run_data

def analyze_function_comparison(vanilla_data, tabpfn_data, problem_id, dimension, optimal_value=None):
    """
    Analyze comparison between algorithms for a specific function and dimension.
    Print summary statistics without plotting.
    """
    if not vanilla_data and not tabpfn_data:
        return None

    # Get optimal value if not provided - estimate from best achieved value
    if optimal_value is None:
        all_best_values = []
        for data in (vanilla_data or []) + (tabpfn_data or []):
            all_best_values.extend(data['best_so_far'])
        optimal_value = min(all_best_values) if all_best_values else 0.0
    
    # Calculate final regret statistics
    if vanilla_data:
        vanilla_regrets = [max(data['best_so_far'][-1] - optimal_value, 1e-12) for data in vanilla_data]
    
    if tabpfn_data:
        tabpfn_regrets = [max(data['best_so_far'][-1] - optimal_value, 1e-12) for data in tabpfn_data]
    
    return True


def run_benchmark():
    """Run the complete benchmark experiment"""
    
    benchmark_manager = BenchmarkManager()
    
    total_experiments = len(BBOB_FUNCTIONS) * len(DIMENSIONS) * len(INSTANCES) * N_REPETITIONS * len(ALGORITHMS)
    completed_experiments = 0
    
    # Run experiments for each dimension
    for dimension in DIMENSIONS:
        dim_dir = benchmark_manager.base_output_dir / f"dim_{dimension}"
        
        # For each function
        for problem_id in BBOB_FUNCTIONS:
            # For each instance
            for instance in INSTANCES:
                # For each repetition
                for repetition in range(1, N_REPETITIONS + 1):
                    # For each algorithm
                    for algorithm in ALGORITHMS:
                        try:
                            result = run_single_optimization(
                                algorithm=algorithm,
                                problem_id=problem_id,
                                dimension=dimension,
                                instance=instance,
                                repetition=repetition,
                                output_dir=dim_dir / algorithm,
                                device=device
                            )
                            
                            completed_experiments += 1
                            
                        except Exception as e:
                            completed_experiments += 1
                    continue
            
            # Create comparison analysis for this function and dimension using .dat files
            if True:  # Always try to analyze data from .dat files
                # Process .dat files from both algorithms
                vanilla_data = []
                tabpfn_data = []
                
                # Read Vanilla BO .dat files
                vanilla_dir = dim_dir / "vanilla"
                if vanilla_dir.exists():
                    dat_files = list(vanilla_dir.rglob("*.dat"))
                    for dat_file in dat_files:
                        try:
                            # Less strict file matching
                            if f"_f{problem_id}_" in dat_file.name:
                                run_data = process_dat_file(dat_file)
                                vanilla_data.append(run_data)
                        except Exception as e:
                            pass
                
                # Read TabPFN BO .dat files  
                tabpfn_dir = dim_dir / "tabpfn"
                if tabpfn_dir.exists():
                    dat_files = list(tabpfn_dir.rglob("*.dat"))
                    for dat_file in dat_files:
                        try:
                            # Less strict file matching
                            if f"_f{problem_id}_" in dat_file.name:
                                run_data = process_dat_file(dat_file)
                                tabpfn_data.append(run_data)
                        except Exception as e:
                            pass
                
                if vanilla_data or tabpfn_data:
                    # Get optimal value from problem instance
                    try:
                        temp_problem = get_problem(problem_id, instance=1, dimension=dimension)
                        optimal_value = temp_problem.optimum.y
                    except Exception as e:
                        optimal_value = None
                    
                    analyze_function_comparison(
                        vanilla_data=vanilla_data,
                        tabpfn_data=tabpfn_data,
                        problem_id=problem_id,
                        dimension=dimension,
                        optimal_value=optimal_value
                    )
            
            # Progress update - simplified
            progress = (completed_experiments / total_experiments) * 100
    
    # Save experiment summary
    benchmark_manager.experiment_log["end_time"] = datetime.now().isoformat()
    benchmark_manager.experiment_log["total_experiments"] = total_experiments
    benchmark_manager.experiment_log["completed_experiments"] = completed_experiments
    
    log_file = benchmark_manager.base_output_dir / "experiment_log.json"
    with open(log_file, 'w') as f:
        json.dump(benchmark_manager.experiment_log, f, indent=2)

if __name__ == "__main__":
    try:
        # Start the benchmark
        run_benchmark()
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {str(e)}")
        traceback.print_exc()