import torch
import torch.nn as nn

class BaseDynamics(nn.Module):
    """
    Base class for dynamics models in PyTorch.
    All specific dynamics implementations should inherit from this class.
    """
    def __init__(self, state_dim, action_dim, dt=0.01):
        """
        Initialize the base dynamics model.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action/control space
            dt (float): Time step for the dynamics
        """
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