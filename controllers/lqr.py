from configs import constants as _C

from typing import Optional, Tuple
from functools import partial

import numpy as np
from scipy.linalg import solve_discrete_are

import functools

from utils.load_traj import get_traj_from_wham, get_traj_from_pkl
from envs.smpl import *
from utils.utils import *

import mujoco

from pdb import set_trace as st

import matplotlib.pyplot as plt

def fhlqr(A: np.array, 
          B: np.array, 
          Q: np.array, 
          R: np.array, 
          Qf: np.array, 
          N: int=100) -> tuple[list, list]:
    nx, nu = B.shape
    assert A.shape == (nx, nx), "A must be of shape (nx, nx)"
    assert Q.shape == (nx, nx), "Q must be of shape (nx, nx)"
    assert R.shape == (nu, nu), "R must be of shape (nu, nu)"
    assert Qf.shape == (nx, nx), "Qf must be of shape (nx, nx)"

    P = [np.zeros((nx, nx)) for _ in range(N)]
    K = [np.zeros((nu, nx)) for _ in range(N - 1)]

    P[-1] = Qf.copy()

    for k in range(N - 2, -1, -1):
        _Q=Q #if k%5==0 else Qf
        K[k] = np.linalg.inv(R + B.T @ P[k + 1] @ B) @ (B.T @ P[k + 1] @ A)
        P[k] = _Q + A.T @ P[k + 1] @ A - A.T @ P[k + 1] @ B @ K[k]

    return P, K

def ihlqr(A: np.array, 
          B: np.array, 
          Q: np.array, 
          R: np.array, 
          tol: float = 1e-5, 
          verbose: bool = False, 
          freq: int = 1000) -> tuple[np.array, np.array]:
    nx, nu = B.shape
    P = Q.copy()

    i=0
    while True:
        if verbose and i%freq==0: print(f"Iteration : {i}")
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K

        if np.linalg.norm(P - P_new, ord=np.inf) < tol:
            if verbose: print(f"Converged at iteration {i}")
            return P_new, K
        i+=1

        P = P_new

if __name__ == '__main__':
    env = SMPL()
    state = env.reset()

    # rot, ang, transl, vel=get_traj_from_wham()
    rot, ang, transl, vel=get_traj_from_pkl()

    root_rot=axis_angle_to_quaternion(rot[:, 0])
    # root_rot=axis_angle_to_quaternion(rot[:, 0]-np.array([0, np.pi/2, 0]))
    # root_rot=axis_angle_to_quaternion(np.zeros_like(rot[:, 0]))
    rot=np.concatenate((root_rot, rot[:, 1:].reshape(root_rot.shape[0], -1)), axis=-1)
    # rot=np.concatenate((root_rot, np.zeros_like(rot[:, 1:].reshape(root_rot.shape[0], -1))), axis=-1)

    ang=ang.reshape(root_rot.shape[0], -1)
    transl=transl+env.initial_pose[:3]

    # ang=np.zeros_like(ang.reshape(root_rot.shape[0], -1))
    # transl=np.zeros_like(transl)+env.initial_pose[:3]

    q_ref=np.concatenate((transl, rot), axis=-1)
    qd_ref=np.concatenate((vel, ang), axis=-1)

    state=env.reset(qpos=q_ref[0], qvel=qd_ref[0])

    dq = np.zeros(env.model.nv)

    nv=env.model.nv
    nu = env.model.nu  # Alias for the number of actuators.
    R = np.eye(nu)

    Q = np.block([[np.eye(nv, nv), np.zeros((nv, nv))],
              [np.zeros((nv, 2*nv))]])
    
    # Example: Randomly wiggle joints
    actions=[]
    for q, qd in zip(q_ref, qd_ref):
        print(len(actions))
        mujoco.mj_differentiatePos(env.model, dq, 1, q, env.data.qpos)
        dx = np.hstack((dq, env.data.qvel-qd)).T

        A, B=linearize_dynamics(env)

        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

        act=- K @ dx

        actions.append(act)
        env.step(act)

        if len(actions)==100:
            break
    
    render_env(env, act=act, qpos=q_ref[0], qvel=qd_ref[0])