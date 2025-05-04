from configs import constants as _C

import pickle
import os
import os.path as osp

import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.interpolate import CubicSpline

from pdb import set_trace as st

def upsample_signal_cubic(original_signal, original_dt=1/40, upsampled_dt=0.003):
    original_time = np.arange(0, len(original_signal) * original_dt, original_dt)
    upsampled_time = np.arange(0, original_time[-1], upsampled_dt)
    cubic_spline = CubicSpline(original_time, original_signal)
    upsampled_signal = cubic_spline(upsampled_time)

    return upsampled_signal

def convert_traj_to_pkl():
    import torch
    result=torch.load('/home/mukundan/Desktop/Summer_SEM/imitation_learning/dataset/data/rep_00_output.pt')#['full_pose'].reshape(-1, 24, 3)
    savefile={key: result[key].numpy() for key in ['pose_embedding', 'full_pose', 'transl', 'betas', 'side_R', 'side_T']}

    with open('/home/mukundan/Desktop/Summer_SEM/imitation_learning/dataset/data/data.pickle', 'wb') as file:
        pickle.dump(savefile, file)

def get_traj_from_pkl():
    with open('test_data/data.pickle', 'rb') as file:
        pose=pickle.load(file)['full_pose'].reshape((-1, 24, 3))

    rot=upsample_signal_cubic(pose.copy())
    ang=upsample_signal_cubic(pose.copy())
    ang=np.concatenate(((ang[1:]-ang[:-1])/40, np.zeros((1, 24, 3))), axis=0)
    # ang=np.concatenate((np.zeros((1, 24, 3)), (ang[1:]-ang[:-1])/40), axis=0)
    transl=np.zeros((ang.shape[0], 3), dtype=float)

    return rot, ang, transl, transl

def get_traj_from_wham(dt=0.003*5):
    results=joblib.load('test_data/drone_video/wham_output.pkl')

    pose=results[0]['body_pose']
    transl=results[0]['trans_world']
    fps=np.ceil(results[0]['fps'])

    root_rot=results[0]['pose_world'][:, :3]
    root_rot=upsample_signal_cubic(root_rot.copy(), original_dt=1/fps, upsampled_dt=dt)[:, [2, 0, 1]]
    root_ang=root_rot.copy()
    root_ang=np.concatenate(((root_ang[1:]-root_ang[:-1])/fps, np.zeros((1, 3))), axis=0)
    
    rot=upsample_signal_cubic(pose.copy(), original_dt=1/fps, upsampled_dt=dt)
    ang=upsample_signal_cubic(pose.copy(), original_dt=1/fps, upsampled_dt=dt)
    ang=np.concatenate(((ang[1:]-ang[:-1])/fps, np.zeros((1, 69))), axis=0)
    transl=upsample_signal_cubic(transl.copy(), original_dt=1/fps, upsampled_dt=dt)[:, [2, 0, 1]]
    # transl=np.column_stack([transl[:, 2], transl[:, 0], np.full(transl.shape[0], transl[:, 1].mean())])
    transl[:, 0]=np.linspace(0, 12, transl.shape[0])
    transl[:, 1]=0
    vel=np.concatenate(((transl[1:]-transl[:-1])/fps, np.zeros((1, 3))), axis=0)

    return rot, ang, transl, vel, root_rot, root_ang

def plot_3d_trajectory(points, title="3D Trajectory", xlabel="X", ylabel="Y", zlabel="Z", show=True, save_path=None):
    points = np.asarray(points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 
            marker='o', markersize=4, linestyle='-', linewidth=1, alpha=0.8)
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], 
               c='green', s=100, label='Start', marker='*')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], 
               c='red', s=100, label='End', marker='*')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend()
    
    ax.set_box_aspect([1, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        
    return fig, ax

if __name__=='__main__':
    rot, ang, transl, vel=get_traj_from_wham()
    plot_3d_trajectory(transl)

    import pdb; pdb.set_trace()

