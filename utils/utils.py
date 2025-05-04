from configs import constants as _C

import numpy as np
import torch
import mujoco

from typing import Tuple, Union

from envs.base import Env
from utils.transforms import *

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

def smpl_to_robot(pose: Union[np.array, torch.tensor]) -> Union[np.array, torch.tensor]:
    return pose[:, list(_C.ROBOT.REVERSE_MAPPING.values())]

def robot_to_smpl(pose: Union[np.array, torch.tensor]) -> Union[np.array, torch.tensor]:
    return pose[:, list(_C.ROBOT.FROM_SMPL_MAP.values())]

def get_actuator_names(model):
    actuators = []
    for i in range(model.nu):
        if i == model.nu - 1:
            end_p = None
            for el in ["name_sensoradr", "name_numericadr", "name_textadr", "name_tupleadr", "name_keyadr", "name_pluginadr"]:
                v = getattr(model, el)
                if np.any(v):
                    end_p = v[0]
            if end_p is None:
                end_p = model.nnames
        else:
            end_p = model.name_actuatoradr[i+1]
        name = model.names[model.name_actuatoradr[i]:end_p].decode("utf-8").rstrip('\x00')
        actuators.append(name)
    return actuators

def linearize_dynamics_for_control(env: Env,
                                   x: np.array,
                                   u: np.array,
                                   epsilon: float=1e-6,
                                   flg_centered: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    model=env.model
    data=env.data

    q=x[:x.shape[0]//2+1]
    qd=x[x.shape[0]//2+1:]

    mujoco.mj_resetData(model, data)
    data.ctrl = u
    data.qpos = q
    data.qvel = qd

    nv=env.model.nv
    nu=env.model.nu

    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    
    mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

    return A, B