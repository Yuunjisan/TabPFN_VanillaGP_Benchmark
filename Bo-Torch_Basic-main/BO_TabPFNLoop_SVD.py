"""
This script visualizes the Bayesian optimization process for 1D functions using TabPFN as a surrogate model
with spatial uncertainty scaling, showing both the TabPFN surrogate model (sausage plots) and acquisition function.

Adapted from Sausage_Acq_plot_BO.py but using TabPFN_BO_Variance instead of regular TabPFN_BO.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import os
from pathlib import Path
import pandas as pd
from scipy import stats

# BoTorch imports
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound

# Import TabPFN_BO_Variance instead of regular TabPFN_BO
from Algorithms.BayesianOptimization.TabPFN_BO.TabPFN_BO_Variance import TabPFN_BO

# Define a simple 1D test function
def f(x):
    """Test function to optimize: f(x) = x^2 * sin(x) + 5"""
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            x = x.ravel()
    return x**2 * np.sin(x) + 5

class TabPFNVarianceVisualizationBO(TabPFN_BO):
    """
    Extension of TabPFN_BO_Variance that adds visualization capabilities for the surrogate model
    and acquisition function during optimization, with spatial uncertainty scaling.
    """
    def __init__(self, budget, n_DoE=4, acquisition_function="expected_improvement", 
                 random_seed=42, save_plots=True, plots_dir="tabpfn_variance_bo_visualizations",
                 n_estimators=8, fit_mode="fit_with_cache", device="cpu", 
                 # Spatial uncertainty parameters
                 spatial_alpha=0.1, spatial_beta=4.0, spatial_sigma=0.3, 
                 enable_spatial_scaling=True, **kwargs):
        super().__init__(budget, n_DoE, acquisition_function, random_seed, 
                         n_estimators=n_estimators, fit_mode=fit_mode, device="cpu",
                         # Pass spatial parameters to parent
                         spatial_alpha=spatial_alpha, spatial_beta=spatial_beta, 
                         spatial_sigma=spatial_sigma, enable_spatial_scaling=enable_spatial_scaling, 
                         **kwargs)
        
        self.device = torch.device('cpu')
        # Visualization settings
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.iteration_count = 0
        self.next_point_tensor = None  # Store the next point for accurate visualization
        
        # Store spatial parameters for visualization
        self.spatial_alpha = spatial_alpha
        self.spatial_beta = spatial_beta
        self.spatial_sigma = spatial_sigma
        self.enable_spatial_scaling = enable_spatial_scaling
        
        # Plot style settings
        self.plot_colors = {
            'true_function': 'r--',
            'tabpfn_mean': 'darkorange',
            'confidence': 'purple',
            'spatial_factors': 'purple',
            'points': 'navy',
            'next_point': 'green',
            'best_point': 'red',
            'acquisition': 'g-'
        }
        
        self.plot_markers = {
            'points': 'o',
            'next_point': '*',
            'best_point': 'x'
        }
        
        # For confidence band visualization
        self.confidence_alphas = [0.05, 0.1, 0.15, 0.2]
        self.quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]  # 10-90%, 20-80%, 30-70%, 40-60%
        
        # Create plots directory
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)
    
    def __call__(self, problem, bounds=None, **kwargs):
        """
        Run Bayesian optimization with visualization at each iteration.
        For 1D problems with visualization of the surrogate model and acquisition function.
        """
        # Initialize the optimization
        self._initialize(problem, bounds, **kwargs)
        
        # Visualize the initial surrogate model
        self.visualize_initial_model()
        
        # Run the optimization loop
        self._run_optimization_loop(problem, **kwargs)
        
        # Final visualization
        self.visualize_final_result()
    
    def _initialize(self, problem, bounds, **kwargs):
        """Initialize the optimization process."""
        # Set default bounds if not provided
        if bounds is None:
            bounds = np.array([[-6.0, 6.0]])  # Wider bounds for showing more of the function
        
        # Force 1D
        dim = 1
        
        # Call the superclass __call__ method with initial sampling
        super(TabPFN_BO, self).__call__(problem, dim, bounds, **kwargs)
        
        # Initialize the TabPFN model
        self._initialize_model()
    
    def _run_optimization_loop(self, problem, **kwargs):
        """Run the main optimization loop with visualization at each step."""
        for cur_iteration in range(self.budget - self.n_DoE):
            self.iteration_count = cur_iteration + 1
            
            # Set up Expected Improvement acquisition function
            self.acquisition_function = self.acquisition_function_class(
                model=self.model,
                best_f=self.current_best,
                maximize=self.maximisation
            )
            
            # Get the next point
            new_x = self.optimize_acqf_and_get_observation()
            
            # Store the selected point for visualization
            self.next_point_tensor = new_x.clone()
            
            # Visualize after we know the next point to evaluate
            self.visualize_iteration()
            
            # Evaluate function - ensure we have a flat vector
            if new_x.dim() > 2:
                new_x = new_x.squeeze(1)  # Remove extra dimension if present
            
            new_x_numpy = new_x.detach().squeeze().numpy()
            
            # Handle both scalar and vector inputs correctly
            if new_x_numpy.ndim == 0:  # Handle scalar case
                new_x_numpy = float(new_x_numpy)
                new_f_eval = float(problem(new_x_numpy))
            else:  # Handle vector case
                new_f_eval = float(problem(new_x_numpy))  # Ensure result is a float
            
            # Append evaluations - ensure consistent data types
            self.x_evals.append(new_x_numpy)
            self.f_evals.append(new_f_eval)
            self.number_of_function_evaluations += 1
            
            # Update model - using in-place update for performance
            X_new = new_x
            Y_new = torch.tensor([new_f_eval], dtype=torch.float64).view(1, 1)
            self.model.condition_on_observations(X=X_new, Y=Y_new, verbose=False)
            
            # Assign new best
            self.assign_new_best()
            
            # Print progress if verbose
            if self.verbose:
                print(f"Iteration {cur_iteration+1}/{self.budget - self.n_DoE}: Best value = {self.current_best}")
        
        print("Optimization Process finalized!")
    
    def _plot_true_function(self, ax, x_range=None):
        """Plot the true function."""
        if x_range is None:
            x_range = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        
        y_true = np.array([f(x_i) for x_i in x_range])
        ax.plot(x_range, y_true, self.plot_colors['true_function'], label='True function')
        return x_range, y_true
    
    def _plot_tabpfn_predictions(self, ax, x_range):
        """Plot TabPFN predictions with uncertainty bands using native BarDistribution quantiles."""
        # Create tensor input for TabPFN
        X_pred = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float64, device=torch.device(self.device))
        
        # Get predictions from TabPFN using both posterior and native quantiles
        with torch.no_grad():
            # 1. Get spatial-aware posterior for mean and variance
            posterior = self.model.posterior(X_pred.unsqueeze(0))  # Add batch dimension
            spatial_mean = posterior.mean.squeeze().detach().cpu().numpy()
            spatial_variance = posterior.variance.squeeze().detach().cpu().numpy()
            spatial_std = np.sqrt(spatial_variance)
            
            # 2. Get native TabPFN quantiles from the underlying model
            # We need to access the TabPFN model directly for quantiles
            predictions = self.model.model.predict(X_pred.cpu().numpy(), output_type="main")
            base_quantiles = predictions["quantiles"]
            base_mean = predictions["mean"]
            
            # 3. Get spatial factors to understand the scaling effect
            if self.enable_spatial_scaling:
                spatial_factors = self.model._compute_spatial_variance_factors(X_pred.unsqueeze(0))
                spatial_factors = spatial_factors.squeeze().detach().cpu().numpy()
            else:
                spatial_factors = np.ones_like(x_range)
            
            # 4. Apply spatial scaling to the base quantiles
            # Calculate the base variance from quantiles for scaling reference
            base_std_approx = (base_quantiles[6] - base_quantiles[2]) / (2 * 1.28)  # Approximate std from 20-80% quantiles
            
            # Scale the quantiles around the base mean using spatial factors
            scaled_quantiles = []
            for i, base_q in enumerate(base_quantiles):
                # Calculate deviation from mean
                deviation = base_q - base_mean
                # Scale the deviation by spatial factors
                scaled_deviation = deviation * np.sqrt(spatial_factors)  # Scale by sqrt since variance scales linearly
                # Add back to spatial-aware mean
                scaled_q = spatial_mean + scaled_deviation
                scaled_quantiles.append(scaled_q)
        
        # Plot spatial-aware mean prediction
        ax.plot(x_range, spatial_mean, color=self.plot_colors['tabpfn_mean'], label='TabPFN mean', linewidth=2)
        
        # Plot spatial-aware median
        median_idx = 4  # Index 4 corresponds to 50th percentile
        ax.plot(x_range, scaled_quantiles[median_idx], color='green', linestyle='--', label='TabPFN median')
        
        # Plot confidence bands using scaled quantiles
        for (lower_idx, upper_idx), alpha in zip(self.quantile_pairs, self.confidence_alphas):
            # Get lower and upper quantiles for this band
            q_low = scaled_quantiles[lower_idx]
            q_high = scaled_quantiles[upper_idx]
            
            # Plot the confidence band
            ax.fill_between(
                x_range, 
                q_low, 
                q_high, 
                alpha=alpha, 
                color=self.plot_colors['confidence'], 
                label=f'Quantile {lower_idx+1}0-{upper_idx+1}0%' if lower_idx == 0 else None
            )
        
        return spatial_mean, scaled_quantiles, spatial_factors
    
    def _plot_spatial_factors(self, ax, x_range):
        """Plot spatial uncertainty factors."""
        if not self.enable_spatial_scaling:
            return np.ones_like(x_range)
        
        # Create tensor input
        X_pred = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float64, device=torch.device(self.device))
        
        # Get spatial factors
        with torch.no_grad():
            spatial_factors = self.model._compute_spatial_variance_factors(X_pred.unsqueeze(0))
            spatial_factors = spatial_factors.squeeze().detach().cpu().numpy()
        
        # Plot spatial factors
        ax.plot(x_range, spatial_factors, color=self.plot_colors['spatial_factors'], 
                linewidth=2, label='Spatial uncertainty factor')
        
        # Add horizontal lines for min/max
        ax.axhline(y=self.spatial_alpha, color='red', linestyle='--', alpha=0.7, 
                  label=f'Minimum (α={self.spatial_alpha})')
        ax.axhline(y=self.spatial_alpha + self.spatial_beta, color='red', linestyle='--', alpha=0.7,
                  label=f'Maximum (α+β={self.spatial_alpha + self.spatial_beta})')
        
        # Highlight training points
        if hasattr(self, 'x_evals') and len(self.x_evals) > 0:
            x_eval = np.array([float(x) if np.isscalar(x) else float(x[0]) for x in self.x_evals])
            ax.scatter(x_eval, [self.spatial_alpha] * len(x_eval), 
                      color='navy', s=50, alpha=0.8, label='Training points')
        
        ax.set_ylabel('Spatial Factor')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return spatial_factors
    
    def _plot_evaluated_points(self, ax, highlight_best=True, highlight_next=None):
        """Plot all evaluated points and optionally highlight the best and/or next point."""
        # Convert points to arrays with consistent shapes
        if len(self.x_evals) > 0:
            # Ensure x_evals are properly shaped
            x_eval = np.array([float(x) if np.isscalar(x) else float(x[0]) for x in self.x_evals])
            # Ensure f_evals are properly shaped
            y_eval = np.array([float(y) for y in self.f_evals])
            
            # Plot all points
            ax.scatter(
                x_eval, y_eval, 
                c=self.plot_colors['points'], 
                marker=self.plot_markers['points'], 
                label='Evaluated points'
            )
            
            # Highlight the best point
            if highlight_best and len(self.x_evals) > 0:
                best_idx = np.argmin(y_eval) if not self.maximisation else np.argmax(y_eval)
                ax.scatter(
                    x_eval[best_idx], y_eval[best_idx], 
                    c=self.plot_colors['best_point'], 
                    marker=self.plot_markers['best_point'], 
                    s=150, 
                    label='Best point'
                )
            
            # Highlight the next point (if provided)
            if highlight_next is not None:
                next_x, next_y = highlight_next
                # Ensure next point values are scalars
                next_x = float(next_x) if np.isscalar(next_x) else float(next_x[0])
                next_y = float(next_y)
                ax.scatter(
                    next_x, next_y, 
                    c=self.plot_colors['next_point'], 
                    marker=self.plot_markers['next_point'], 
                    s=200, 
                    label='Next point'
                )
    
    def _save_plot(self, fig, filename):
        """Save the plot to a file if save_plots is enabled."""
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def visualize_initial_model(self):
        """Visualize the initial surrogate model with the DoE points."""
        # Create a figure for the initial state
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot the true function
        x_range, _ = self._plot_true_function(ax1)
        
        # Plot the initial points
        self._plot_evaluated_points(ax1, highlight_best=True, highlight_next=None)
        
        # If we already have some initial points, we can plot the initial model
        if len(self.x_evals) >= 2:  # Need at least 2 points for a TabPFN model
            # Generate x points for prediction
            x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
            
            # Plot TabPFN predictions
            self._plot_tabpfn_predictions(ax1, x_pred)
        
        # Set title and labels for main plot
        ax1.set_title(f'Initial Design of Experiments (DoE), n={len(self.x_evals)}' + 
                     (f' - Spatial Scaling: α={self.spatial_alpha}, β={self.spatial_beta}, σ={self.spatial_sigma}' 
                      if self.enable_spatial_scaling else ' - No Spatial Scaling'))
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot spatial factors
        if len(self.x_evals) >= 2:
            x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
            self._plot_spatial_factors(ax2, x_pred)
            ax2.set_title('Spatial Uncertainty Factors')
            ax2.set_xlabel('x')
        
        plt.tight_layout()
        self._save_plot(fig, 'initial_doe.png')
    
    def visualize_iteration(self):
        """Visualize the current surrogate model and acquisition function."""
        # Create figure with three subplots: surrogate model, spatial factors, and acquisition function
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 1. Plot the surrogate model (sausage plot)
        # -----------------------------------------
        
        # Generate x points for prediction
        x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        
        # Plot the true function
        self._plot_true_function(ax1)
        
        # Plot TabPFN predictions
        mean, scaled_quantiles, spatial_factors = self._plot_tabpfn_predictions(ax1, x_pred)
        
        # 2. Calculate acquisition function values
        # ----------------------------------------------------------------
        # Evaluate the acquisition function at each x point
        acq_values = []
        for x in x_pred:
            # Reshape for single point evaluation
            x_tensor = torch.tensor([[x]], dtype=torch.float64)
            with torch.no_grad():
                acq_val = self.acquisition_function(x_tensor)
            acq_values.append(acq_val.item())
        
        # Find the maximum of the acquisition function based on our grid evaluation
        max_idx = np.argmax(acq_values)
        grid_max_x = x_pred[max_idx]
        
        # Get the actual next point we'll evaluate (if available)
        if self.next_point_tensor is not None:
            # Convert the next point tensor to a scalar
            next_x_np = self.next_point_tensor.detach().squeeze().cpu().numpy()
            
            # Handle different array dimensions properly
            if np.isscalar(next_x_np) or next_x_np.ndim == 0:
                next_x = float(next_x_np)
            else:
                next_x = float(next_x_np[0])
                
            # Get mean prediction for this point
            closest_idx = np.abs(x_pred - next_x).argmin()
            next_median = scaled_quantiles[4][closest_idx]  # Index 4 corresponds to 50th percentile (median)
        else:
            # If no next point is provided, use the grid maximum
            next_x = grid_max_x
            next_median = scaled_quantiles[4][max_idx]  # Use median instead of mean
        
        # Plot all evaluated points and highlight the best and next points
        self._plot_evaluated_points(ax1, highlight_best=True, highlight_next=(next_x, next_median))
        
        # Set title and labels for the surrogate plot
        title = f'Iteration {self.iteration_count}: TabPFN Surrogate Model with Spatial Scaling'
        if self.enable_spatial_scaling:
            title += f' (α={self.spatial_alpha}, β={self.spatial_beta}, σ={self.spatial_sigma})'
        else:
            title += ' (Spatial scaling disabled)'
        ax1.set_title(title)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # 2. Plot spatial factors
        # ----------------------
        # Reuse spatial factors already computed in _plot_tabpfn_predictions for efficiency
        if self.enable_spatial_scaling:
            ax2.plot(x_pred, spatial_factors, color=self.plot_colors['spatial_factors'], 
                    linewidth=2, label='Spatial uncertainty factor')
            
            # Add horizontal lines for min/max
            ax2.axhline(y=self.spatial_alpha, color='red', linestyle='--', alpha=0.7, 
                      label=f'Minimum (α={self.spatial_alpha})')
            ax2.axhline(y=self.spatial_alpha + self.spatial_beta, color='red', linestyle='--', alpha=0.7,
                      label=f'Maximum (α+β={self.spatial_alpha + self.spatial_beta})')
            
            # Highlight training points
            if hasattr(self, 'x_evals') and len(self.x_evals) > 0:
                x_eval = np.array([float(x) if np.isscalar(x) else float(x[0]) for x in self.x_evals])
                ax2.scatter(x_eval, [self.spatial_alpha] * len(x_eval), 
                          color='navy', s=50, alpha=0.8, label='Training points')
            
            ax2.set_ylabel('Spatial Factor')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Spatial scaling disabled', 
                   transform=ax2.transAxes, ha='center', va='center',
                   fontsize=14, fontweight='bold')
        
        ax2.set_title('Spatial Uncertainty Factors')
        ax2.set_xlabel('x')
        
        # 3. Plot the acquisition function
        # --------------------------------
        
        # Plot the actual acquisition function values
        acq_array = np.array(acq_values)
        ax3.plot(x_pred, acq_array, self.plot_colors['acquisition'], linewidth=2, label='Acquisition function')
        
        # If we have an actual next point from the optimizer, mark it
        if self.next_point_tensor is not None:
            # Find the closest grid point for a clean visualization
            closest_idx = np.abs(x_pred - next_x).argmin()
            ax3.scatter(
                next_x, acq_array[closest_idx], 
                c=self.plot_colors['next_point'], 
                marker=self.plot_markers['next_point'], 
                s=200, 
                label='Next point'
            )
        else:
            # Just mark the grid maximum
            ax3.scatter(
                x_pred[max_idx], acq_array[max_idx], 
                c=self.plot_colors['next_point'], 
                marker=self.plot_markers['next_point'], 
                s=200, 
                label='Next point'
            )
        
        ax3.set_ylabel('Acquisition value')
        ax3.legend(loc='best')
        ax3.set_title(f'Acquisition Function ({self.acquistion_function_name})')
        ax3.set_xlabel('x')
        ax3.grid(True)
        
        plt.tight_layout()
        
        self._save_plot(fig, f'iteration_{self.iteration_count:03d}.png')
    
    def visualize_final_result(self):
        """Visualize the final surrogate model."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Generate x points for prediction
        x_pred = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        
        # Plot the true function
        self._plot_true_function(ax1)
        
        # Plot TabPFN predictions
        self._plot_tabpfn_predictions(ax1, x_pred)
        
        # Plot all evaluated points and highlight the best point
        self._plot_evaluated_points(ax1, highlight_best=True)
        
        # Set title and labels
        title = f'Final TabPFN Surrogate Model with Spatial Scaling after {self.budget} evaluations'
        if self.enable_spatial_scaling:
            title += f'\n(α={self.spatial_alpha}, β={self.spatial_beta}, σ={self.spatial_sigma})'
        else:
            title += '\n(Spatial scaling disabled)'
        ax1.set_title(title)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # Plot final spatial factors
        self._plot_spatial_factors(ax2, x_pred)
        ax2.set_title('Final Spatial Uncertainty Factors')
        ax2.set_xlabel('x')
        
        plt.tight_layout()
        
        self._save_plot(fig, 'final_model.png')

def run_1d_tabpfn_variance_visualization_bo(acquisition_function="expected_improvement", budget=15, 
                                           n_DoE=3, bounds=None, save_plots=True, n_estimators=8, 
                                           temperature=1.25, 
                                           # Spatial uncertainty parameters
                                           spatial_alpha=0.1, spatial_beta=4.0, spatial_sigma=0.3,
                                           enable_spatial_scaling=True):
    """
    Run Bayesian optimization with TabPFN spatial variance visualization on a 1D test function.
    
    Args:
        acquisition_function: Which acquisition function to use
        budget: Total function evaluation budget
        n_DoE: Number of initial points
        bounds: Problem bounds (default: [-6.0, 6.0])
        save_plots: Whether to save plots to disk
        n_estimators: Number of TabPFN estimators
        temperature: Softmax temperature for TabPFN (affects uncertainty quantification)
        spatial_alpha: Minimum variance multiplier at observed points
        spatial_beta: Maximum additional variance for distant points
        spatial_sigma: Length scale for distance-based scaling
        enable_spatial_scaling: Whether to enable spatial uncertainty scaling
    """
    # Set default bounds
    if bounds is None:
        bounds = np.array([[-6.0, 6.0]])  # Wider bounds for showing more of the function
    
    spatial_status = "enabled" if enable_spatial_scaling else "disabled"
    print(f"Starting TabPFN Variance Bayesian Optimization visualization with {acquisition_function}")
    print(f"Spatial scaling: {spatial_status} (α={spatial_alpha}, β={spatial_beta}, σ={spatial_sigma})")
    
    try:
        # Create directory for plots
        spatial_suffix = f"_spatial_α{spatial_alpha}_β{spatial_beta}_σ{spatial_sigma}" if enable_spatial_scaling else "_no_spatial"
        plots_dir = f"tabpfn_variance_bo_visualizations/{acquisition_function}{spatial_suffix}"
        
        # Set up the TabPFN-based Bayesian optimizer with visualization
        optimizer = TabPFNVarianceVisualizationBO(
            budget=budget,
            n_DoE=n_DoE,
            acquisition_function=acquisition_function,
            random_seed=42,
            verbose=True,
            maximisation=False,
            save_plots=save_plots,
            plots_dir=plots_dir,
            n_estimators=n_estimators,
            temperature=temperature,
            # Spatial uncertainty parameters
            spatial_alpha=spatial_alpha,
            spatial_beta=spatial_beta,
            spatial_sigma=spatial_sigma,
            enable_spatial_scaling=enable_spatial_scaling,
            device="cpu"  # Force CPU
        )
        
        # Run optimization with visualization
        optimizer(
            problem=f,
            bounds=bounds
        )
        
        print(f"Optimization completed successfully.")
        print(f"Best point found: x = {float(optimizer.x_evals[optimizer.current_best_index]) if np.isscalar(optimizer.x_evals[optimizer.current_best_index]) else float(optimizer.x_evals[optimizer.current_best_index][0]):.4f}, f(x) = {float(optimizer.current_best):.4f}")
        
    except Exception as e:
        print(f"ERROR: Run failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run with spatial scaling enabled using the specified parameters
    print("🔬 Testing TabPFN with Spatial Uncertainty Scaling")
    run_1d_tabpfn_variance_visualization_bo(
        acquisition_function="expected_improvement",
        budget=20,
        n_DoE=3,  # Use fewer initial points to see spatial scaling effect
        n_estimators=8,
        temperature=0.9,
        # Spatial uncertainty parameters as specified
        spatial_alpha=0.1,
        spatial_beta=5.0,
        spatial_sigma=0.1,
        enable_spatial_scaling=True
    )
    
    # Also run with spatial scaling disabled for comparison
    print("\n🔬 Testing TabPFN without Spatial Uncertainty Scaling (for comparison)")
    run_1d_tabpfn_variance_visualization_bo(
        acquisition_function="expected_improvement",
        budget=20,
        n_DoE=3,
        n_estimators=8,
        temperature=1.25,
        # Disable spatial scaling
        spatial_alpha=0.1,
        spatial_beta=4.0,
        spatial_sigma=0.3,
        enable_spatial_scaling=False
    )