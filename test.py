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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

smpl = build_smpl_model(device, model_type='smpl', use_vposer=False)

with torch.no_grad():
    pose_params = torch.zeros(1, 69).to(device)  # 23 joints * 3 (SMPL standard)
    pose_params[0]=torch.pi/2
    smpl.reset_params(body_pose=pose_params)

output = smpl(return_verts=True)
vertices = output.vertices[0]

camera = build_cameras(smpl, is_sideview=False)
renderer = SMPLRenderer(camera, smpl.model.faces, device)

rendered_img = renderer.render_mesh(vertices)

output_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('smpl_output.png', output_img)
print(f"Image saved to smpl_output.png")

env=SMPL()
_=env.reset()
env_renderer=EnvRenderer(env)
env_image=env_renderer.render_image()

cv2.imwrite('env_output.png', env_image)
print(f"Image saved to env_output.png")

stacked_img = np.vstack((output_img, env_image))

cv2.imshow('SMPL Model', stacked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()