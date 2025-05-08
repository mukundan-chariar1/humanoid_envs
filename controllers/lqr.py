from configs import constants as _C

from typing import Optional, Tuple
from functools import partial

import numpy as np
from scipy.linalg import solve_discrete_are

import cv2

import functools

from utils.load_traj import get_traj_from_wham
from envs.smpl import *
from utils.utils import *
from utils.transforms import *
from utils.viz import Renderer

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

def main():
    env = SMPL()
    state = env.reset()
    env_renderer=Renderer(env)

    rot, ang, transl_, vel=get_traj_from_wham()

    root_rot=axis_angle_to_quaternion(np.zeros((rot.shape[0], 3)))
    rot=np.concatenate((root_rot, smpl_to_robot(rot)), axis=-1)
    ang=np.concatenate((np.zeros((root_rot.shape[0], 3)), smpl_to_robot(ang)), axis=-1)
    transl=transl_+env.initial_pose[:3]

    q_ref=np.concatenate((transl, rot), axis=-1)
    qd_ref=np.concatenate((vel, ang), axis=-1)

    state=env.reset(qpos=q_ref[0], qvel=qd_ref[0])

    dq = np.zeros(env.model.nv)

    nv=env.model.nv
    nu = env.model.nu  # Alias for the number of actuators.
    R = np.eye(nu)*10

    Q = np.block([[np.eye(nv, nv), np.zeros((nv, nv))],
              [np.zeros((nv, 2*nv))]])
    
    # Example: Randomly wiggle joints
    actions=[]
    for q, qd in zip(q_ref[1:], qd_ref[1:]):
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
    
    env_renderer.render_env(act=act, qpos=q_ref[0], qvel=qd_ref[0])

if __name__ == '__main__':
    env = SMPL()
    state = env.reset()
    env_renderer=Renderer(env)

    rot, ang, transl_, vel, root_rot, root_ang = get_traj_from_wham(env.dt)

    root_rot = axis_angle_to_quaternion(np.zeros_like(root_rot))
    rot = np.concatenate((root_rot, smpl_to_robot(rot)), axis=-1)
    ang = np.concatenate((np.zeros_like(root_ang), smpl_to_robot(ang)), axis=-1)

    transl_[:, 2]=0
    transl = transl_ + env.initial_pose[:3]

    q_ref = np.concatenate((transl, rot), axis=-1)
    qd_ref = np.concatenate((vel, ang), axis=-1)
    xref = np.concatenate([q_ref, qd_ref], axis=-1)[:300]

    state=env.reset(qpos=q_ref[0], qvel=qd_ref[0])

    dq = np.zeros(env.model.nv)

    nv=env.model.nv
    nu = env.model.nu  # Alias for the number of actuators.
    R = np.eye(nu)*10

    Q = np.block([[np.eye(nv, nv), np.zeros((nv, nv))],
              [np.zeros((nv, 2*nv))]])
    
    # Example: Randomly wiggle joints
    actions=[]
    x_positions=[q_ref[0, 2]]
    for q, qd in zip(q_ref[1:], qd_ref[1:]):
        print(len(actions))
        mujoco.mj_differentiatePos(env.model, dq, 1, q, env.data.qpos)
        dx = np.hstack((dq, env.data.qvel-qd)).T

        A, B=linearize_dynamics(env)

        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

        act=- K @ dx

        actions.append(act)
        env.step(act)

        x_positions.append(env.data.qpos[2])

        if len(actions)==100:
            break

    import matplotlib.pyplot as plt

    # Common x-axis value
    x = np.full((100, 1), 1.4)
    # Two different y-values
    y = np.stack(x_positions)

    y=0.21+(y-y.min())/(y.max()-y.min())*(1.4-0.21)

    # Plot
    plt.plot(x, c='b', label='reference trajectory')
    plt.plot(y, c='r', label='generated trajectory')
    plt.title('Reference versus Generated Trajectory of Pelvis \nfor first 100 steps of simulation')
    plt.xlabel('Z position')
    plt.ylabel('Number of simulation steps')
    plt.grid(True)
    plt.savefig('ref_vs_gen.png')
    plt.show()

    

    import pdb; pdb.set_trace()


    
    
    env_renderer.render_env(act=act, qpos=q_ref[0], qvel=qd_ref[0])