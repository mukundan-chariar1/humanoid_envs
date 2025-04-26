import numpy as np
import mujoco

from typing import Tuple

from envs.base import Env

def linearize_dynamics(env: Env,
                       epsilon: float=1e-6,
                       flg_centered: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    model=env.model
    data=env.data

    nv=env.model.nv
    nu=env.model.nu

    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    
    mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

    return A, B
