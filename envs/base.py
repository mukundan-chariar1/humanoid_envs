import abc
from typing import Any, Dict, Tuple, Optional

class Env(abc.ABC):
    """Abstract base class for environment interfaces."""
    
    def __init__(self):
        """Initialize the environment."""
        super().__init__()
        self._is_done = False
        self._current_state = None
    
    @abc.abstractmethod
    def reset(self) -> Any:
        """Reset the environment to its initial state.
        
        Returns:
            The initial observation/state of the environment.
        """
        pass
    
    @abc.abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Run one timestep of the environment's dynamics.
        
        Args:
            action: The action to take in the environment.
            
        Returns:
            A tuple containing:
                - next_state: The new state after the action
                - reward: The reward from taking the action
                - done: Whether the episode has ended
                - info: Additional diagnostic information
        """
        pass
    
    @property
    @abc.abstractmethod
    def action_space(self) -> Any:
        """Returns the action space of the environment."""
        pass
    
    @property
    @abc.abstractmethod
    def observation_space(self) -> Any:
        """Returns the observation space of the environment."""
        pass


from typing import Any, Dict, Optional, Hashable
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    """
    A generic state class that can be used to represent system states.
    
    Attributes:
        id: A unique identifier for the state (hashable)
        data: The actual state data (can be any type)
        metadata: Additional information about the state (dict)
        is_terminal: Whether this is a terminal/absorbing state
    """
    id: Hashable
    data: Any
    metadata: Dict[str, Any] = None
    is_terminal: bool = False
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def __hash__(self) -> int:
        """Make the state hashable based on its ID."""
        return hash(self.id)
    
    def __eq__(self, other: 'State') -> bool:
        """Compare states based on their IDs."""
        if not isinstance(other, State):
            return False
        return self.id == other.id
    
    def to_numpy(self) -> np.ndarray:
        """Convert state data to numpy array if possible."""
        if isinstance(self.data, np.ndarray):
            return self.data
        try:
            return np.array(self.data)
        except:
            raise ValueError("State data cannot be converted to numpy array")
    
    def copy(self) -> 'State':
        """Create a deep copy of the state."""
        return State(
            id=self.id,
            data=self.data.copy() if hasattr(self.data, 'copy') else self.data,
            metadata=self.metadata.copy(),
            is_terminal=self.is_terminal
        )
    
    def update(self, **kwargs) -> None:
        """Update state attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.metadata[key] = value

    @property
    def shape(self) -> tuple:
        """Get the shape of the state data (if array-like)."""
        if hasattr(self.data, 'shape'):
            return self.data.shape
        if isinstance(self.data, (list, tuple)):
            return (len(self.data),)
        return (1,)