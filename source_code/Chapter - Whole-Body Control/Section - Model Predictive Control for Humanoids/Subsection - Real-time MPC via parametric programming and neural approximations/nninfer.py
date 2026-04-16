import torch
import numpy as np
from typing import Union, Tuple

# Load pre-trained model for control prediction
model = torch.load('mpqp_surrogate.pt')
model.eval()

def online_control(x: np.ndarray, u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    """
    Compute control input for given state using learned surrogate model.
    
    Args:
        x: State vector as numpy array
        u_min: Minimum control bounds
        u_max: Maximum control bounds
        
    Returns:
        Projected control input satisfying box constraints
    """
    # Convert numpy input to torch tensor for model inference
    x_t = torch.tensor(x, dtype=torch.float32)
    
    # Forward pass through surrogate model to get candidate control
    with torch.no_grad():
        u_hat = model(x_t).detach().numpy()
    
    # Project to feasible control set using box constraints
    u_proj = np.clip(u_hat, u_min, u_max)
    
    return u_proj