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
    A simplified BoTorch-compatible wrapper for TabPFN, closely following TabPFN_BOTORCH.py.
    """
    def __init__(
        self,
        train_X: np.ndarray,
        train_Y: np.ndarray,
        n_estimators: int = 8,
        temperature: float = 1.25,
        fit_mode: str = "fit_with_cache",
        device: str = "auto"
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
        
        # Initialize and fit TabPFN
        self.model = TabPFNRegressor(
            n_estimators=n_estimators, 
            softmax_temperature=temperature,
            fit_mode=fit_mode,
            device=self.device
        )
        self.model.fit(train_X, train_Y.ravel())
    
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
        Returns the posterior at X using TabPFN's native BarDistribution mean and variance.
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
            variance = criterion.variance(logits).detach().cpu().numpy()  # Shape: [n_points]
            
            # Ensure positive variance - use reasonable minimum for EI exploration
            variance = np.maximum(variance, 1e-4)  # Increased minimum for better EI exploration
            std = np.sqrt(variance)
            
        except Exception as e:
            print(f"Error in TabPFN BarDistribution prediction: {e}")
            # Fallback with reasonable uncertainty for EI exploration
            sample_size = X_reshaped.shape[0]
            mean = np.zeros(sample_size)
            std = np.ones(sample_size) * 0.1  # Reasonable std for EI exploration
        
        # Convert to tensors with correct shape matching X's dtype and device
        target_device = torch.device(self.device)
        mean_tensor = torch.tensor(mean, dtype=X.dtype, device=target_device)
        std_tensor = torch.tensor(std, dtype=X.dtype, device=target_device)
        
        # Reshape output for BoTorch
        # For batched case, we need to ensure the output has shape [batch_shape, q, 1]
        if len(batch_shape) > 0:
            # First reshape back to match the batch structure
            output_shape = batch_shape + (q, 1)
            mean_out = mean_tensor.view(output_shape)
            std_out = std_tensor.view(output_shape)
        else:
            # For non-batched case, reshape to [q, 1]
            mean_out = mean_tensor.view(q, 1)
            std_out = std_tensor.view(q, 1)
        
        # Create a dummy tensor connected to X's computation graph for gradient tracking
        dummy = X.sum() * 0
        
        # Connect tensors to input graph
        mean_out = mean_out + dummy
        std_out = std_out + dummy
        
        # Create MVN distribution with appropriate dimension handling
        return MultivariateNormal(
            mean_out,
            scale_tril=std_out.unsqueeze(-1)  # Make scale_tril have shape [..., q, 1, 1]
        )
    
    def condition_on_observations(self, X: Tensor, Y: Tensor, verbose: bool = False, **kwargs) -> Model:
        """Update the model with new observations.
        
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
        new_train_X = torch.cat([self.train_X, X_flat], dim=0).cpu().numpy()
        new_train_Y = torch.cat([self.train_Y.view(-1), Y_flat], dim=0).cpu().numpy()
        
        # Update this model in-place
        if verbose:
            print(f"Updating TabPFN model in-place: {len(self.train_X)} -> {len(new_train_X)} points")
        
        self.train_X = torch.tensor(new_train_X, dtype=torch.float64, device=target_device)
        self.train_Y = torch.tensor(new_train_Y, dtype=torch.float64, device=target_device).view(-1, 1)
        
        # Re-fit the TabPFN model
        self.model.fit(new_train_X, new_train_Y)
        return self


class TabPFN_BO(AbstractBayesianOptimizer):
    def __init__(self, budget, n_DoE=0, acquisition_function:str="expected_improvement",
                 random_seed:int=43, n_estimators:int=8, temperature:float=1.25, 
                 fit_mode:str="fit_with_cache", device:str="auto", **kwargs):
        """
        Bayesian Optimization using TabPFN as a surrogate model.
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
        
        # Set up the acquisition function
        self.__acq_func = None
        self.acquistion_function_name = acquisition_function
        
        # TabPFN surrogate model
        self.__model_obj = None
        
    def __str__(self):
        return "This is an instance of TabPFN BO Optimizer"

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
        
        self.__model_obj = TabPFNSurrogateModel(
            train_X=train_x,
            train_Y=train_obj,
            n_estimators=self.n_estimators,
            temperature=self.temperature,
            fit_mode=self.fit_mode,
            device=self.device
        )
    
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