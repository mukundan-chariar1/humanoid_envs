from configs import constants as _C

from envs import *
import mujoco
import numpy as np
from typing import Dict, Any, Optional, Tuple

from utils.viz import *

class SMPL(Env):
    """MuJoCo environment for SMPL human model."""
    
    def __init__(self, 
                 num_simulation_step_per_control_step: int=5):
        super().__init__()
        self.xml_path = 'models/smpl.xml'
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # SMPL-specific parameters
        self.body_names = self._get_body_names()
        self.joint_names = self._get_joint_names()
        self.initial_pose = self._get_default_pose()
        self.initial_vel = self._get_default_vel()

        self.num_simulation_step_per_control_step=num_simulation_step_per_control_step
        self.dt=num_simulation_step_per_control_step*self.model.opt.timestep
        
    def _get_body_names(self) -> list:
        """Extract all body names from the model."""
        return [self.model.body(i).name for i in range(self.model.nbody)]
    
    def _get_joint_names(self) -> list:
        """Extract all joint names from the model."""
        return [self.model.joint(i).name for i in range(self.model.njnt)]
    
    def _get_default_pose(self) -> np.ndarray:
        """Return the default SMPL pose (T-pose)."""
        return self.data.qpos
    
    def _get_default_vel(self) -> np.ndarray:
        """Return the default SMPL pose (T-pose)."""
        return self.data.qvel
    
    def _get_body_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all bodies."""
        return {name: self.data.body(name).xpos.copy() for name in self.body_names}
    
    def reset(self, 
              qpos: Optional[np.array] = None, 
              qvel: Optional[np.array] = None) -> Dict[str, Any]:
        """Reset the SMPL model to default pose."""
        mujoco.mj_resetData(self.model, self.data)
        if qpos is None:
            self.data.qpos[:] = self.initial_pose
            self.data.qvel[:] = self.initial_vel
        else:
            self.data.qpos[:] = qpos
            if qvel is not None: self.data.qvel[:] = qvel
            else: self.data.qvel[:] = self.initial_vel
        mujoco.mj_forward(self.model, self.data)
        
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'body_positions': self._get_body_positions()
        }
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        # Apply control (direct joint position control for SMPL)
        self.data.ctrl[:] = action
        
        # Step the simulation
        for i in range(self.num_simulation_step_per_control_step): mujoco.mj_step(self.model, self.data)
        
        # Get new state
        obs = {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'body_positions': self._get_body_positions()
        }
        
        # For SMPL, we might not have a traditional reward
        reward = 0.0
        done = False
        info = {'action_applied': action.copy()}
        
        return obs, reward, done, info
    
    def action_space(self):
        return self.model.nu
    
    def observation_space(self):
        return self.model.nv+self.model.nq

if __name__ == '__main__':
    env = SMPL()
    state = env.reset()
    
    # Example: Randomly wiggle joints
    act=[]
    for _ in range(100):
        random_action = np.random.uniform(-1, 1, size=env.model.nu)
        obs, _, _, _ = env.step(random_action)

        act.append(random_action)
    
    render_env(env, act=act)