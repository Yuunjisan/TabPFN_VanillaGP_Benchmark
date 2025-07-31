"""
TabPFN Regression Visualization - Optimized

Visualizes TabPFN regression on mathematical functions over domain [-2, 2]:
1. sin(5x) + x, 2. x², 3. Multi-step function, 4. |x|

To change sample points, modify the SAMPLE_POINTS dictionary below.
"""

import numpy as np
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Configuration: Adjust sample points for each function
SAMPLE_POINTS = {
    "sin_plus_x": 37,    # sin(5x) + x
    "sphere": 7,        # x²
    "step": 37,          # Step function: 4 plateaus × 11 points each
    "abs": 7           # |x|
}

def generate_data(func_name, n_samples, x_range=(-0.5, 0.5)):
    if func_name == "step":
        # Distribute n_samples points across 4 plateaus
        x = []
        y = []
        plateau_boundaries = [-0.5, -0.25, 0, 0.25, 0.5]
        plateau_values = [-0.4, -0.2, 0.2, 0.4]
        
        # Calculate points per plateau
        points_per_plateau = n_samples // 4
        remaining_points = n_samples % 4
        
        for i in range(4):  # 4 plateaus
            # Add extra point to first plateaus if there are remaining points
            current_points = points_per_plateau + (1 if i < remaining_points else 0)
            
            # Generate points evenly spaced within each plateau
            x_plateau = np.linspace(plateau_boundaries[i], plateau_boundaries[i+1], current_points, endpoint=(i==3))
            y_plateau = np.full(current_points, plateau_values[i])
            x.extend(x_plateau)
            y.extend(y_plateau)
        
        return np.array(x), np.array(y)
    else:
        x = np.linspace(x_range[0], x_range[1], n_samples)
        # Scale x by 4 to compress the function horizontally (same as true_function)
        x_scaled = 4 * x
        
        if func_name == "sin_plus_x":
            y = np.sin(5 * x_scaled) + x_scaled
        elif func_name == "sphere":
            y = x_scaled**2
        elif func_name == "abs":
            y = np.abs(x_scaled)
        return x, y

def true_function(func_name, x):
    if func_name == "step":
        y = np.zeros_like(x)
        y = np.where(x < -0.25, -0.4, y)
        y = np.where((x >= -0.25) & (x < 0), -0.2, y)
        y = np.where((x >= 0) & (x < 0.25), 0.2, y)
        y = np.where(x >= 0.25, 0.4, y)
        return y
    else:
        # Scale x by 4 to compress the function horizontally
        x_scaled = 4 * x
        if func_name == "sin_plus_x":
            return np.sin(5 * x_scaled) + x_scaled
        elif func_name == "sphere":
            return x_scaled**2
        elif func_name == "abs":
            return np.abs(x_scaled)

def visualize_function(func_name, x_train, y_train, title):
    model = TabPFNRegressor(n_estimators=8, softmax_temperature=0.9, random_state=42)
    model.fit(x_train.reshape(-1, 1), y_train)
    
    x_test = np.linspace(-0.5, 0.5, 200)
    predictions = model.predict(x_test.reshape(-1, 1), output_type="main", quantiles=[0.1, 0.25, 0.75, 0.9])
    
    y_pred_mean = predictions["mean"]
    y_pred_median = predictions["median"]
    quantiles = predictions["quantiles"]
    y_10th, y_25th, y_75th, y_90th = quantiles
    y_true = true_function(func_name, x_test)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_test, y_true, 'k-', linewidth=2, label='True Function', alpha=0.8)
    plt.scatter(x_train, y_train, c='red', s=50, alpha=0.7, label=f'Training Data (n={len(x_train)})', zorder=5)
    plt.plot(x_test, y_pred_mean, 'b-', linewidth=2, label='TabPFN Mean')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'{title}\nTabPFN Regression with Uncertainty', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{func_name}_{len(x_train)}pts.png", dpi=300, bbox_inches='tight')
    plt.show()

# Generate visualizations
functions = [
    ("sin_plus_x", "sin(5x) + x"),
    ("sphere", "1D Sphere Function x²"),
    ("step", "Multi-Step Function"),
    ("abs", "Absolute Value |x|")
]

for func_name, title in functions:
    n_samples = SAMPLE_POINTS[func_name]
    x_train, y_train = generate_data(func_name, n_samples)
    visualize_function(func_name, x_train, y_train, title) 