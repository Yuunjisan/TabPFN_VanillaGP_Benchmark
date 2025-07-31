import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs
import os
import torch
import matplotlib.gridspec as gridspec
import sys

# Add the correct path to import the algorithms
sys.path.append(os.path.dirname(__file__))

# Now import using the correct path structure
from Algorithms.BayesianOptimization.TabPFN_BO.TabPFN_BO_Variance import TabPFN_BO, TabPFNSurrogateModel

# Set random seed for reproducibility
np.random.seed(42)

def f(X:np.ndarray):
    x = X.copy()
    return x**2 * np.sin(x) + 5

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def generate_tabpfn_variance_plot(n_samples, softmax_temp, n_estimators, save_dir, 
                                 spatial_alpha=0.3, spatial_beta=2.0, spatial_sigma=0.4,
                                 enable_spatial_scaling=True, xi=None, fi=None, y_min=None, y_max=None):
    """Generate and save a TabPFN Variance plot with spatial uncertainty scaling"""
    
    # Define bounds
    lb = -6.0
    ub = 6.0

    # Use provided samples or generate new ones
    if xi is None or fi is None:
        # Sample points using Latin Hypercube Sampling
        np.random.seed(42)
        samples_normalized = lhs(1, samples=n_samples, criterion="center")
        xi = lb + (ub - lb) * samples_normalized
        fi = f(xi).ravel()

    # Create bounds array for the TabPFN surrogate model
    bounds = np.array([[lb, ub]])

    # Create and fit the TabPFN surrogate model with spatial uncertainty scaling
    model = TabPFNSurrogateModel(
        train_X=xi,
        train_Y=fi,
        n_estimators=n_estimators,
        temperature=softmax_temp,
        fit_mode="fit_with_cache",
        device="auto",
        spatial_alpha=spatial_alpha,
        spatial_beta=spatial_beta,
        spatial_sigma=spatial_sigma,
        enable_spatial_scaling=enable_spatial_scaling,
        bounds=bounds
    )

    # Define points for predictions (fine grid for smooth visualization)
    X_pred = np.atleast_2d(np.linspace(lb, ub, 100)).T
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float64, device=torch.device(model.device))

    # Get predictions using both the surrogate model's posterior and native quantiles
    with torch.no_grad():
        # 1. Get spatial-aware posterior for mean and variance
        posterior = model.posterior(X_pred_tensor.unsqueeze(0))  # Add batch dimension
        spatial_mean = posterior.mean.squeeze().detach().cpu().numpy()
        spatial_variance = posterior.variance.squeeze().detach().cpu().numpy()
        spatial_std = np.sqrt(spatial_variance)
        
        # 2. Get native TabPFN quantiles from the underlying model
        predictions = model.model.predict(X_pred, output_type="main")
        base_quantiles = predictions["quantiles"]
        base_mean = predictions["mean"]
        
        # 3. Get spatial factors to understand the scaling effect
        if enable_spatial_scaling:
            spatial_factors = model._compute_spatial_variance_factors(X_pred_tensor.unsqueeze(0))
            spatial_factors = spatial_factors.squeeze().detach().cpu().numpy()
        else:
            spatial_factors = np.ones_like(X_pred.ravel())
        
        # 4. Apply spatial scaling to the base quantiles
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
    
    # Define quantile pairs and alphas for confidence bands (matching BO_TabPFNLoop style)
    confidence_alphas = [0.05, 0.1, 0.15, 0.2]
    quantile_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]  # 10-90%, 20-80%, 30-70%, 40-60%
    
    # Get spatial uncertainty factors for visualization
    if enable_spatial_scaling:
        spatial_factors = model._compute_spatial_variance_factors(X_pred_tensor.unsqueeze(0))
        spatial_factors = spatial_factors.squeeze().detach().cpu().numpy()
    else:
        spatial_factors = np.ones_like(X_pred.ravel())
    
    # ============================================================================
    # CREATE SINGLE FIGURE WITH PROPER GRID LAYOUT
    # ============================================================================
    
    # Calculate figure dimensions
    fig_width = 15
    fig_height = 6
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create GridSpec: 1 row, 2 columns
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
    
    # ============================================================================
    # LEFT: MAIN PREDICTION PLOT
    # ============================================================================
    ax_main = fig.add_subplot(gs[0, 0])
    
    # Plot the true function
    X_true = np.atleast_2d(np.linspace(lb, ub, 500)).T
    ax_main.plot(X_true, f(X_true), 'r--', label='True function', linewidth=2)

    # Plot confidence bands using scaled quantiles (matching BO_TabPFNLoop style)
    for (lower_idx, upper_idx), alpha in zip(quantile_pairs, confidence_alphas):
        # Get lower and upper quantiles for this band
        q_low = scaled_quantiles[lower_idx]
        q_high = scaled_quantiles[upper_idx]
        
        # Plot the confidence band
        ax_main.fill_between(
            X_pred.ravel(), 
            q_low, 
            q_high, 
            alpha=alpha, 
            color='purple', 
            label=f'Quantile {lower_idx+1}0-{upper_idx+1}0%' if lower_idx == 0 else None
        )

    # Plot the predicted mean
    ax_main.plot(X_pred, spatial_mean, color='darkorange', label='TabPFN mean', linewidth=2)
    
    # Plot the predicted median (50th percentile)
    median_idx = 4  # Index 4 corresponds to 50th percentile
    ax_main.plot(X_pred, scaled_quantiles[median_idx], color='green', linestyle='--', label='TabPFN median')
    
    # Plot training data
    ax_main.scatter(xi, fi, color='navy', s=50, alpha=0.8, 
                   label='Training points')

    ax_main.set_xlabel('x', fontsize=12)
    ax_main.set_ylabel('f(x)', fontsize=12)
    title = f'TabPFN with Spatial Uncertainty Scaling\n(n={n_samples}, temp={softmax_temp}, estimators={n_estimators})'
    if enable_spatial_scaling:
        title += f'\nα={spatial_alpha}, β={spatial_beta}, σ={spatial_sigma}'
    else:
        title += '\n(Spatial scaling disabled)'
    ax_main.set_title(title, fontsize=12, fontweight='bold')
    
    ax_main.legend(fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(labelsize=10)
    
    # Set y-axis limits if provided
    if y_min is not None and y_max is not None:
        ax_main.set_ylim(y_min, y_max)
    
    # ============================================================================
    # RIGHT: SPATIAL UNCERTAINTY FACTORS
    # ============================================================================
    ax_spatial = fig.add_subplot(gs[0, 1])
    
    if enable_spatial_scaling:
        # Plot spatial uncertainty factors
        ax_spatial.plot(X_pred, spatial_factors, color='purple', linewidth=2, label='Spatial uncertainty factor')
        ax_spatial.axhline(y=spatial_alpha, color='red', linestyle='--', alpha=0.7, 
                          label=f'Minimum (α={spatial_alpha})')
        ax_spatial.axhline(y=spatial_alpha + spatial_beta, color='red', linestyle='--', alpha=0.7,
                          label=f'Maximum (α+β={spatial_alpha + spatial_beta})')
        
        # Highlight training points
        ax_spatial.scatter(xi, [spatial_alpha] * len(xi), color='navy', s=50, alpha=0.8,
                          label='Training points')
        
        ax_spatial.set_xlabel('x', fontsize=12)
        ax_spatial.set_ylabel('Spatial Uncertainty Factor', fontsize=12)
        ax_spatial.set_title('Spatial Uncertainty Scaling', fontsize=12, fontweight='bold')
        ax_spatial.legend(fontsize=9)
        ax_spatial.grid(True, alpha=0.3)
        ax_spatial.tick_params(labelsize=10)
    else:
        ax_spatial.text(0.5, 0.5, 'Spatial scaling disabled', 
                       transform=ax_spatial.transAxes, ha='center', va='center',
                       fontsize=14, fontweight='bold')
        ax_spatial.set_title('Spatial Uncertainty Scaling', fontsize=12, fontweight='bold')
    
    # ============================================================================
    # SAVE FIGURE
    # ============================================================================
    plt.tight_layout()
    
    # Save the plot
    spatial_str = f"_spatial_α{spatial_alpha}_β{spatial_beta}_σ{spatial_sigma}" if enable_spatial_scaling else "_no_spatial"
    base_filename = f"tabpfn_variance_samples_{n_samples}_temp_{softmax_temp:.2f}_estimators_{n_estimators}{spatial_str}"
    filename = f"{base_filename}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return model

# Create plots directory
plots_dir = ensure_dir("tabpfn_variance_plots")

# Define sample sizes and parameters to test
sample_sizes = [6, 8, 10]  # Use fewer points to create gaps for spatial scaling
temperatures = [0.9]
n_estimators = [8]

# Spatial uncertainty parameters to test - with more dramatic scaling
spatial_configs = [
    {"alpha": 0.1, "beta": 5.0, "sigma": 0.1, "enabled": True},   # Small sigma for dramatic scaling
    {"alpha": 0.3, "beta": 2.0, "sigma": 0.4, "enabled": False},  # No spatial scaling for comparison
]

# Calculate y-axis limits based on the true function with a 20% margin
lb = -6.0
ub = 6.0
X_true = np.atleast_2d(np.linspace(lb, ub, 500)).T
y_true = f(X_true)
y_min_val = y_true.min()
y_max_val = y_true.max()
y_range = y_max_val - y_min_val
margin = 0.2 * y_range
y_min = y_min_val - margin
y_max = y_max_val + margin

# Generate plots for each combination
for n_samples in sample_sizes:
    # Generate samples once for each sample size
    np.random.seed(42)
    samples_normalized = lhs(1, samples=n_samples, criterion="center")
    xi = lb + (ub - lb) * samples_normalized
    fi = f(xi).ravel()
    
    for temp in temperatures:
        for n_est in n_estimators:
            for config in spatial_configs:
                generate_tabpfn_variance_plot(
                    n_samples=n_samples, 
                    softmax_temp=temp, 
                    n_estimators=n_est, 
                    save_dir=plots_dir,
                    spatial_alpha=config["alpha"],
                    spatial_beta=config["beta"], 
                    spatial_sigma=config["sigma"],
                    enable_spatial_scaling=config["enabled"],
                    xi=xi, 
                    fi=fi, 
                    y_min=y_min, 
                    y_max=y_max
                )

print(f"All TabPFN Variance plots saved to '{plots_dir}' directory.")
