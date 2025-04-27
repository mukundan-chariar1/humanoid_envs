from configs import constants as _C

import cv2
import torch

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np

from smpl_lib.smpl import build_smpl_model
from smpl_lib.camera import build_cameras
from smpl_lib.viz import Renderer as SMPLRenderer

from envs.smpl import SMPL
from utils.viz import Renderer as EnvRenderer

from utils.load_traj import get_traj_from_wham

from pytorch3d import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pose, transl=get_traj_from_wham()

smpl = build_smpl_model(device, model_type='smpl', use_vposer=False)

with torch.no_grad():
    pose_params = torch.from_numpy(pose[100:101]).to(device) # 23 joints * 3 (SMPL standard)
    
    pose_params_r6d=transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(pose_params.reshape(1, -1, 3)))

    smpl.pose_embedding.index_copy_(
                0, torch.LongTensor(range(len(smpl.pose_embedding))
            ).to(device), pose_params_r6d.to(device))

output = smpl(return_verts=True)
vertices = output.vertices[0]

camera = build_cameras(smpl, is_sideview=False)
renderer = SMPLRenderer(camera, smpl.model.faces, device)

rendered_img = renderer.render_mesh(vertices)

output_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('smpl_output.png', output_img)
print(f"Image saved to smpl_output.png")

env=SMPL()

qpos=env.initial_pose.copy()
qpos[list(_C.ROBOT.FROM_SMPL_MAP.values())]=pose_params.cpu().numpy().squeeze()
_=env.reset()
env_renderer=EnvRenderer(env)
env_image=env_renderer.render_image(qpos=qpos)

cv2.imwrite('env_output.png', env_image)
print(f"Image saved to env_output.png")

stacked_img = np.vstack((output_img, env_image))

cv2.imshow('SMPL Model', stacked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

env_renderer.close()