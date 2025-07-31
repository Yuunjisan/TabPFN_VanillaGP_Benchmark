from ..AbstractBayesianOptimizer import AbstractBayesianOptimizer
from typing import Union, Callable, Optional, List, Dict, Any, Tuple
from ioh.iohcpp.problem import RealSingleObjective
import numpy as np
import torch
import os
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition.analytic import ExpectedImprovement, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.posteriors import Posterior
from torch.distributions import MultivariateNormal
from tabpfn import TabPFNRegressor


# Only Expected Improvement is supported

class TabPFNSurrogateModel(Model):
    """
    A BoTorch-compatible wrapper for TabPFN with spatial uncertainty scaling.
    Implements distance-based variance modification to improve exploration-exploitation balance.
    """
    def __init__(
        self,
        train_X: np.ndarray,
        train_Y: np.ndarray,
        n_estimators: int = 8,
        temperature: float = 1.25,
        fit_mode: str = "fit_with_cache",
        device: str = "auto",
        # Spatial uncertainty parameters
        spatial_alpha: float = 0.3,    # minimum variance multiplier at observed points
        spatial_beta: float = 2.0,     # maximum additional variance for distant points  
        spatial_sigma: float = 0.4,    # length scale for distance-based scaling
        bounds: Optional[np.ndarray] = None,  # bounds for coordinate normalization
        enable_spatial_scaling: bool = True   # enable/disable spatial scaling
    ):
        super().__init__()
        
        # Determine device
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        device_obj = torch.device(self.device)
        
        # Convert numpy arrays to tensors on the correct device
        self.train_X = torch.tensor(train_X, dtype=torch.float64, device=device_obj)
        self.train_Y = torch.tensor(train_Y, dtype=torch.float64, device=device_obj).view(-1, 1)
        self.n_estimators = n_estimators
        self.temperature = temperature
        self.fit_mode = fit_mode
        
        # Spatial uncertainty parameters
        self.spatial_alpha = spatial_alpha
        self.spatial_beta = spatial_beta 
        self.spatial_sigma = spatial_sigma
        self.enable_spatial_scaling = enable_spatial_scaling
        
        # Store bounds for coordinate normalization
        if bounds is not None:
            bounds_array = np.array(bounds)
            # Convert from standard BO format [d, 2] to our format [2, d]
            if bounds_array.shape[1] == 2:  # Standard format: [[min1, max1], [min2, max2], ...]
                lower_bounds = bounds_array[:, 0]  # [d]
                upper_bounds = bounds_array[:, 1]  # [d]
                self.bounds = torch.stack([
                    torch.tensor(lower_bounds, dtype=torch.float64, device=device_obj),
                    torch.tensor(upper_bounds, dtype=torch.float64, device=device_obj)
                ], dim=0)  # shape: [2, d]
            else:  # Assume already in our format [2, d]
                self.bounds = torch.tensor(bounds, dtype=torch.float64, device=device_obj)
        else:
            # Estimate bounds from training data with some padding
            if len(train_X) > 0:
                data_min = torch.min(self.train_X, dim=0)[0]
                data_max = torch.max(self.train_X, dim=0)[0]
                data_range = data_max - data_min
                padding = 0.1 * data_range  # 10% padding
                self.bounds = torch.stack([
                    data_min - padding,
                    data_max + padding
                ], dim=0)  # shape: [2, d]
            else:
                # Default bounds for empty dataset
                d = train_X.shape[1] if len(train_X.shape) > 1 else 1
                self.bounds = torch.tensor([[-1.0] * d, [1.0] * d], 
                                         dtype=torch.float64, device=device_obj)
        
        # Initialize and fit TabPFN
        self.model = TabPFNRegressor(
            n_estimators=n_estimators, 
            softmax_temperature=temperature,
            fit_mode=fit_mode,
            device=self.device
        )
        self.model.fit(train_X, train_Y.ravel())
    
    def _normalize_coordinates(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [0, 1] using stored bounds."""
        # X shape: [..., d]
        # bounds shape: [2, d] where bounds[0] = lower, bounds[1] = upper
        
        lower_bounds = self.bounds[0]  # shape: [d]
        upper_bounds = self.bounds[1]  # shape: [d] 
        
        # Avoid division by zero
        range_vals = upper_bounds - lower_bounds
        range_vals = torch.where(range_vals < 1e-10, torch.ones_like(range_vals), range_vals)
        
        # Normalize to [0, 1]
        X_normalized = (X - lower_bounds) / range_vals
        return X_normalized
    
    def _compute_spatial_variance_factors(self, X_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial variance scaling factors based on distance to nearest observed points.
        
        Args:
            X_pred: Prediction points, shape [..., d]
            
        Returns:
            Variance scaling factors, shape [..., 1]
        """
        if not self.enable_spatial_scaling or len(self.train_X) == 0:
            # Return uniform scaling if spatial scaling is disabled or no training data
            return torch.ones(X_pred.shape[:-1] + (1,), dtype=X_pred.dtype, device=X_pred.device)
        
        # Flatten X_pred for distance computation
        original_shape = X_pred.shape
        X_pred_flat = X_pred.view(-1, original_shape[-1])  # [n_pred, d]
        
        # Normalize coordinates to [0, 1] for consistent distance scaling
        X_pred_norm = self._normalize_coordinates(X_pred_flat)  # [n_pred, d]
        X_train_norm = self._normalize_coordinates(self.train_X)  # [n_train, d]
        
        # Compute pairwise distances: [n_pred, n_train]
        distances = torch.cdist(X_pred_norm, X_train_norm, p=2)
        
        # Find minimum distance to any observed point: [n_pred]
        min_distances, _ = torch.min(distances, dim=1)
        
        # Apply RBF-like scaling: f(d) = α + β(1 - exp(-d²/2σ²))
        # This gives minimum uncertainty α at observed points, maximum α + β far away
        scaling_factors = self.spatial_alpha + self.spatial_beta * (
            1.0 - torch.exp(-min_distances**2 / (2 * self.spatial_sigma**2))
        )
        
        # Reshape back to match original input shape
        scaling_factors = scaling_factors.view(original_shape[:-1] + (1,))
        
        return scaling_factors

    @property
    def num_outputs(self) -> int:
        return 1
        
    @property
    def batch_shape(self):
        return torch.Size([])
    
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        **kwargs
    ) -> Posterior:
        """
        Returns the posterior at X using TabPFN's native BarDistribution mean and variance,
        with spatial uncertainty scaling based on distance to observed points.
        X shape: [batch_shape, q, d] where q is typically 1 for acquisition functions
        """
        # Extract original batch shape and q value for proper reshaping
        batch_shape = X.shape[:-2]  # Everything before the last two dimensions
        q = X.shape[-2]  # Number of points per batch
        d = X.shape[-1]  # Dimension of each point
        
        # Reshape X for TabPFN input (samples, features) while preserving batch structure
        if len(batch_shape) > 0:
            # For batched case
            batch_size = torch.prod(torch.tensor(batch_shape)).item()
            X_reshaped = X.view(batch_size * q, d).detach().cpu().numpy()
        else:
            # For non-batched case
            X_reshaped = X.view(q, d).detach().cpu().numpy()
        
        # Get predictions from TabPFN using full output to access BarDistribution
        try:
            # Use output_type="full" to get logits and criterion (BarDistribution)
            output = self.model.predict(X_reshaped, output_type="full")
            logits = output["logits"]
            criterion = output["criterion"]  # This is the BarDistribution
            
            # Extract mean and variance directly from BarDistribution
            mean = criterion.mean(logits).detach().cpu().numpy()  # Shape: [n_points]
            variance_base = criterion.variance(logits).detach().cpu().numpy()  # Shape: [n_points]
            
            # Ensure positive variance - use reasonable minimum for EI exploration
            variance_base = np.maximum(variance_base, 1e-6)  # Minimum baseline variance
            
        except Exception as e:
            print(f"Error in TabPFN BarDistribution prediction: {e}")
            # Fallback with reasonable uncertainty for EI exploration
            sample_size = X_reshaped.shape[0]
            mean = np.zeros(sample_size)
            variance_base = np.ones(sample_size) * 1e-4  # Reasonable baseline variance
        
        # Convert to tensors with correct shape matching X's dtype and device
        target_device = torch.device(self.device)
        mean_tensor = torch.tensor(mean, dtype=X.dtype, device=target_device)
        variance_base_tensor = torch.tensor(variance_base, dtype=X.dtype, device=target_device)
        
        # Compute spatial variance scaling factors
        spatial_factors = self._compute_spatial_variance_factors(X)  # shape: [batch_shape, q, 1]
        
        # Apply spatial scaling to variance
        if len(batch_shape) > 0:
            # For batched case: reshape spatial factors to match variance tensor
            spatial_factors_flat = spatial_factors.view(-1)  # [batch_size * q]
            variance_scaled = variance_base_tensor * spatial_factors_flat
            
            # Reshape to final output shape
            output_shape = batch_shape + (q, 1)
            mean_out = mean_tensor.view(output_shape)
            variance_out = variance_scaled.view(output_shape)
        else:
            # For non-batched case
            spatial_factors_flat = spatial_factors.view(-1)  # [q]
            variance_scaled = variance_base_tensor * spatial_factors_flat
            
            mean_out = mean_tensor.view(q, 1)
            variance_out = variance_scaled.view(q, 1)
        
        # Convert variance to standard deviation
        std_out = torch.sqrt(variance_out)
        
        # Create a dummy tensor connected to X's computation graph for gradient tracking
        dummy = X.sum() * 0
        
        # Connect tensors to input graph - CRITICAL for gradient flow
        mean_out = mean_out + dummy
        std_out = std_out + dummy
        
        # Create MVN distribution with appropriate dimension handling
        return MultivariateNormal(
            mean_out,
            scale_tril=std_out.unsqueeze(-1)  # Make scale_tril have shape [..., q, 1, 1]
        )
    
    def condition_on_observations(self, X: Tensor, Y: Tensor, verbose: bool = False, **kwargs) -> Model:
        """Update the model with new observations and recompute bounds if needed.
        
        Args:
            X: New input points
            Y: New observations
            verbose: If True, print update information
            **kwargs: Additional parameters
            
        Returns:
            Updated model (self)
        """
        # Get the target device
        target_device = torch.device(self.device)
        
        # Handle various possible tensor shapes for X and Y
        if X.dim() > 2:
            # Handle batched inputs: squeeze out the q dimension
            X_to_add = X.squeeze(-2)
        else:
            X_to_add = X
            
        # Make sure Y is properly shaped
        if Y.dim() > 2:
            Y_to_add = Y.squeeze(-1)
        else:
            Y_to_add = Y

        # Create a flat view of the tensors and move to target device
        X_flat = X_to_add.reshape(-1, X_to_add.shape[-1]).to(target_device)
        Y_flat = Y_to_add.reshape(-1).to(target_device)
        
        # Ensure existing training data is on the same device
        self.train_X = self.train_X.to(target_device)
        self.train_Y = self.train_Y.to(target_device)
        
        # Combine with existing training data
        new_train_X = torch.cat([self.train_X, X_flat], dim=0)
        new_train_Y = torch.cat([self.train_Y.view(-1), Y_flat], dim=0)
        
        # Update bounds to include new data points
        if self.enable_spatial_scaling:
            data_min = torch.min(new_train_X, dim=0)[0]
            data_max = torch.max(new_train_X, dim=0)[0]
            data_range = data_max - data_min
            padding = 0.1 * data_range  # 10% padding
            self.bounds = torch.stack([
                data_min - padding,
                data_max + padding
            ], dim=0).to(target_device)
        
        # Update this model in-place
        if verbose:
            print(f"Updating TabPFN model in-place: {len(self.train_X)} -> {len(new_train_X)} points")
            if self.enable_spatial_scaling:
                print(f"Updated bounds: {self.bounds[0].cpu().numpy()} to {self.bounds[1].cpu().numpy()}")
        
        self.train_X = new_train_X
        self.train_Y = new_train_Y.view(-1, 1)
        
        # Re-fit the TabPFN model
        self.model.fit(new_train_X.cpu().numpy(), new_train_Y.cpu().numpy())
        return self


class TabPFN_BO(AbstractBayesianOptimizer):
    def __init__(self, budget, n_DoE=0, acquisition_function:str="expected_improvement",
                 random_seed:int=43, n_estimators:int=8, temperature:float=1.25, 
                 fit_mode:str="fit_with_cache", device:str="auto", 
                 # Spatial uncertainty parameters
                 spatial_alpha:float=0.3, spatial_beta:float=2.0, spatial_sigma:float=0.4,
                 enable_spatial_scaling:bool=True, **kwargs):
        """
        Bayesian Optimization using TabPFN as a surrogate model with spatial uncertainty scaling.
        Only Expected Improvement acquisition function is supported.
        
        Args:
            budget: Total number of function evaluations allowed
            n_DoE: Number of initial design points
            acquisition_function: Only "expected_improvement" or "EI" is supported
            random_seed: Random seed for reproducibility
            n_estimators: Number of TabPFN estimators in the ensemble
            temperature: Softmax temperature for TabPFN (affects uncertainty quantification)
            fit_mode: Mode for TabPFN fitting, one of "low_memory", "fit_preprocessors", or "fit_with_cache"
            device: Device to use ("cpu", "cuda", or "auto" for automatic detection)
            spatial_alpha: Minimum variance multiplier at observed points (default: 0.3)
            spatial_beta: Maximum additional variance for distant points (default: 2.0)
            spatial_sigma: Length scale for distance-based scaling (default: 0.4)
            enable_spatial_scaling: Whether to enable spatial uncertainty scaling (default: True)
            **kwargs: Additional parameters
        """
        # Call the superclass
        super().__init__(budget, n_DoE, random_seed, **kwargs)
        
        # Determine device
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # TabPFN specific parameters
        self.n_estimators = n_estimators
        self.temperature = temperature  # Store temperature parameter
        self.fit_mode = fit_mode
        
        # Spatial uncertainty parameters
        self.spatial_alpha = spatial_alpha
        self.spatial_beta = spatial_beta
        self.spatial_sigma = spatial_sigma
        self.enable_spatial_scaling = enable_spatial_scaling
        
        # Set up the acquisition function
        self.__acq_func = None
        self.acquistion_function_name = acquisition_function
        
        # TabPFN surrogate model
        self.__model_obj = None
        
    def update_spatial_parameters(self, spatial_alpha: Optional[float] = None, 
                                  spatial_beta: Optional[float] = None,
                                  spatial_sigma: Optional[float] = None,
                                  enable_spatial_scaling: Optional[bool] = None):
        """Update spatial uncertainty scaling parameters.
        
        Args:
            spatial_alpha: New minimum variance multiplier (if provided)
            spatial_beta: New maximum additional variance (if provided) 
            spatial_sigma: New length scale (if provided)
            enable_spatial_scaling: Enable/disable spatial scaling (if provided)
        """
        if spatial_alpha is not None:
            self.spatial_alpha = spatial_alpha
            if self.__model_obj is not None:
                self.__model_obj.spatial_alpha = spatial_alpha
        
        if spatial_beta is not None:
            self.spatial_beta = spatial_beta
            if self.__model_obj is not None:
                self.__model_obj.spatial_beta = spatial_beta
                
        if spatial_sigma is not None:
            self.spatial_sigma = spatial_sigma
            if self.__model_obj is not None:
                self.__model_obj.spatial_sigma = spatial_sigma
                
        if enable_spatial_scaling is not None:
            self.enable_spatial_scaling = enable_spatial_scaling
            if self.__model_obj is not None:
                self.__model_obj.enable_spatial_scaling = enable_spatial_scaling
                
        print(f"Updated spatial parameters: α={self.spatial_alpha}, β={self.spatial_beta}, σ={self.spatial_sigma}, enabled={self.enable_spatial_scaling}")
        
    def __str__(self):
        if self.enable_spatial_scaling:
            return f"TabPFN BO Optimizer with Spatial Uncertainty Scaling (α={self.spatial_alpha}, β={self.spatial_beta}, σ={self.spatial_sigma})"
        else:
            return "TabPFN BO Optimizer (spatial scaling disabled)"

    def __call__(self, problem:Union[RealSingleObjective,Callable], 
                 dim:Optional[int]=-1, 
                 bounds:Optional[np.ndarray]=None, 
                 **kwargs)-> None:
        """
        Run Bayesian optimization using TabPFN as surrogate model.
        """
        # Call the superclass to run the initial sampling of the problem
        super().__call__(problem, dim, bounds, **kwargs)
        
        # Initialize model once
        self._initialize_model()
        
        # Start the optimization loop
        for cur_iteration in range(self.budget - self.n_DoE):
            # Set up Expected Improvement acquisition function
            self.acquisition_function = self.acquisition_function_class(
                model=self.__model_obj,
                best_f=self.current_best,
                maximize=self.maximisation
            )
            
            # Get next point
            new_x = self.optimize_acqf_and_get_observation()
            
            # Evaluate function - ensure proper dimensionality
            if new_x.dim() > 2:
                new_x = new_x.squeeze(1)  # Remove extra dimension if present
            
            # Properly extract numpy array maintaining all dimensions
            new_x_numpy = new_x.detach().cpu().numpy()
            
            # Handle scalar vs vector correctly without dimension reduction
            if new_x_numpy.ndim == 1 and new_x_numpy.size == 1:  # Handle 1D scalar case
                new_x_numpy = float(new_x_numpy[0])
                new_f_eval = float(problem(new_x_numpy))
            elif new_x_numpy.ndim == 2 and new_x_numpy.shape[0] == 1:  # Handle single row vector
                new_x_numpy = new_x_numpy.reshape(-1)  # Convert to 1D array
                new_f_eval = float(problem(new_x_numpy))
            else:  # Handle general vector case
                new_f_eval = float(problem(new_x_numpy.squeeze()))  # Ensure result is a float
            
            # Append evaluations - ensure consistent data types
            self.x_evals.append(new_x_numpy)
            self.f_evals.append(new_f_eval)
            self.number_of_function_evaluations += 1
            
            # Update model - now using in-place update for performance
            X_new = new_x
            Y_new = torch.tensor([new_f_eval], dtype=torch.float64).view(1, 1)
            # Using in-place update significantly improves performance by avoiding recreating the model
            self.__model_obj.condition_on_observations(X=X_new, Y=Y_new, verbose=self.verbose)
            
            # Assign new best
            self.assign_new_best()
            
            # Print progress
            if self.verbose:
                print(f"Iteration {cur_iteration+1}/{self.budget - self.n_DoE}: Best value = {self.current_best}")
        
        print("Optimization finished!")

    def assign_new_best(self):
        # Call the super class
        super().assign_new_best()
    
    def _initialize_model(self):
        """Initialize the TabPFN surrogate model with current data."""
        # Create properly shaped arrays with consistent dimensions
        # FIXED: Properly handle multi-dimensional inputs without dimensionality reduction
        if len(self.x_evals) > 0:
            # Handle the case where inputs might be mixed scalar and array values
            if np.isscalar(self.x_evals[0]) or (isinstance(self.x_evals[0], np.ndarray) and self.x_evals[0].size == 1):
                # For scalar inputs, reshape to column vector
                train_x = np.array([float(x) for x in self.x_evals]).reshape((-1, 1))
            else:
                # For vector inputs, preserve all dimensions
                train_x = np.array([x if isinstance(x, np.ndarray) else np.array(x) for x in self.x_evals])
                # Ensure consistent shape
                if train_x.ndim == 1:
                    train_x = train_x.reshape(-1, 1)
        else:
            # Empty dataset case
            train_x = np.array([]).reshape(0, self.dimension)
            
        train_obj = np.array([float(y) for y in self.f_evals]).reshape(-1)
        
        # Create bounds from problem bounds if available
        bounds_for_model = self.bounds if hasattr(self, 'bounds') and self.bounds is not None else None
        
        self.__model_obj = TabPFNSurrogateModel(
            train_X=train_x,
            train_Y=train_obj,
            n_estimators=self.n_estimators,
            temperature=self.temperature,
            fit_mode=self.fit_mode,
            device=self.device,
            # Pass spatial uncertainty parameters to the surrogate model
            spatial_alpha=self.spatial_alpha,
            spatial_beta=self.spatial_beta,
            spatial_sigma=self.spatial_sigma,
            enable_spatial_scaling=self.enable_spatial_scaling,
            # Pass bounds to the surrogate model
            bounds=bounds_for_model
        )
    
    def get_spatial_uncertainty_diagnostics(self, X_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get diagnostic information about spatial uncertainty scaling.
        
        Args:
            X_test: Optional test points to evaluate. If None, uses a grid over the bounds.
            
        Returns:
            Dictionary with diagnostic information
        """
        if self.__model_obj is None or not self.enable_spatial_scaling:
            return {"spatial_scaling_enabled": False}
        
        # Create test points if not provided
        if X_test is None:
            if hasattr(self, 'bounds') and self.bounds is not None:
                # Create a 1D grid for visualization
                if self.dimension == 1:
                    # self.bounds is in standard BO format [d, 2]
                    x_min, x_max = self.bounds[0, 0], self.bounds[0, 1]
                    X_test = np.linspace(x_min, x_max, 100).reshape(-1, 1)
                else:
                    # For higher dimensions, create points around observed data
                    if len(self.x_evals) > 0:
                        train_x = np.array([x if isinstance(x, np.ndarray) else np.array([x]) for x in self.x_evals])
                        center = np.mean(train_x, axis=0)
                        std = np.std(train_x, axis=0)
                        X_test = np.random.normal(center, std, (50, self.dimension))
                    else:
                        return {"spatial_scaling_enabled": True, "error": "No training data available"}
            else:
                return {"spatial_scaling_enabled": True, "error": "No bounds available"}
        
        # Convert to tensor
        X_tensor = torch.tensor(X_test, dtype=torch.float64, device=torch.device(self.device))
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get spatial factors
        spatial_factors = self.__model_obj._compute_spatial_variance_factors(X_tensor)
        spatial_factors_np = spatial_factors.squeeze().cpu().numpy()
        
        # Get base variance for comparison
        try:
            posterior = self.__model_obj.posterior(X_tensor)
            base_variance = posterior.variance.squeeze().cpu().numpy()
        except:
            base_variance = None
        
        diagnostics = {
            "spatial_scaling_enabled": True,
            "spatial_alpha": self.spatial_alpha,
            "spatial_beta": self.spatial_beta, 
            "spatial_sigma": self.spatial_sigma,
            "test_points": X_test,
            "spatial_factors": spatial_factors_np,
            "spatial_factor_range": (float(np.min(spatial_factors_np)), float(np.max(spatial_factors_np))),
            "base_variance": base_variance,
            "n_training_points": len(self.x_evals)
        }
        
        return diagnostics
    
    def optimize_acqf_and_get_observation(self):
        """Optimizes the acquisition function and returns a new candidate."""
        device = torch.device(self.device)
        bounds_tensor = torch.tensor(self.bounds.transpose(), dtype=torch.float64, device=device)
        
        try:
            # Print diagnostic info
            print(f"Attempting to optimize acquisition function: {self.__acquisition_function_name}")
            
            candidates, acq_value = optimize_acqf(
                acq_function=self.acquisition_function,
                bounds=bounds_tensor,
                q=1,
                num_restarts=10,
                raw_samples=512, 
            )
            
            # Ensure shape is correct (n x d tensor where n=1)
            if candidates.dim() == 3:
                candidates = candidates.squeeze(1)
            
            print(f"Acquisition function optimization SUCCEEDED with value: {acq_value.item():.6f}")
            return candidates
            
        except Exception as e:
            print(f"ERROR: Acquisition function optimization FAILED: {e}")
            print("Falling back to random sampling!")
            
            # Fallback to random sampling - ensure correct shape with proper dimensionality
            lb = self.bounds[:, 0]
            ub = self.bounds[:, 1]
            random_point = lb + np.random.rand(self.dimension) * (ub - lb)
            return torch.tensor(random_point.reshape(1, self.dimension), dtype=torch.float64, device=device)
    
    def set_acquisition_function_subclass(self) -> None:
        """Set the acquisition function class - only Expected Improvement is supported"""
        if self.__acquisition_function_name == "expected_improvement":
            self.__acq_func_class = ExpectedImprovement
        else:
            raise ValueError(f"Only Expected Improvement is supported, got: {self.__acquisition_function_name}")
    
    def __repr__(self):
        return super().__repr__()
    
    def reset(self):
        return super().reset()
    
    @property
    def acquistion_function_name(self) -> str:
        return self.__acquisition_function_name
    
    @acquistion_function_name.setter
    def acquistion_function_name(self, new_name: str) -> None:
        # Remove some spaces and convert to lower case
        new_name = new_name.strip().lower()
        
        # Only allow Expected Improvement or its shorthand
        if new_name == "ei":
            self.__acquisition_function_name = "expected_improvement"
        elif new_name == "expected_improvement":
            self.__acquisition_function_name = "expected_improvement"
        else:
            raise ValueError(f"Only Expected Improvement ('expected_improvement' or 'EI') is supported, got: {new_name}")
        
        # Set up the acquisition function subclass
        self.set_acquisition_function_subclass()
    
    @property
    def acquisition_function_class(self) -> Callable:
        return self.__acq_func_class
    
    @property
    def acquisition_function(self) -> AnalyticAcquisitionFunction:
        """Returns the acquisition function"""
        return self.__acq_func
    
    @acquisition_function.setter
    def acquisition_function(self, new_acquisition_function: AnalyticAcquisitionFunction) -> None:
        """Sets the acquisition function"""
        if issubclass(type(new_acquisition_function), AnalyticAcquisitionFunction):
            self.__acq_func = new_acquisition_function
        else:
            raise AttributeError("Cannot assign the acquisition function as this does not inherit from the class `AnalyticAcquisitionFunction`",
                                name="acquisition_function",
                                obj=self.__acq_func)
    
    @property
    def model(self):
        """Get the TabPFN surrogate model."""
        return self.__model_obj 