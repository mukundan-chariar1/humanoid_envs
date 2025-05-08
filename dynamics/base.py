import torch
import torch.nn as nn

import abc
from typing import Any, Dict, Tuple, Optional

class BaseDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, dt=0.01):
        super(BaseDynamics, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt
        
    def forward(self, state, action):
        """
        Compute the next state given current state and action.
        
        Args:
            state (torch.Tensor): Current state [batch_size, state_dim]
            action (torch.Tensor): Applied action [batch_size, action_dim]
            
        Returns:
            torch.Tensor: Next state [batch_size, state_dim]
        """
        raise NotImplementedError("Forward method must be implemented by subclass")
        
    def derivative(self, state, action):
        """
        Compute the time derivative of the state (dx/dt).
        
        Args:
            state (torch.Tensor): Current state [batch_size, state_dim]
            action (torch.Tensor): Applied action [batch_size, action_dim]
            
        Returns:
            torch.Tensor: Time derivative of state [batch_size, state_dim]
        """
        raise NotImplementedError("Derivative method must be implemented by subclass")
    
class BaseTransform(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self):
        pass

    def inverse(self):
        pass