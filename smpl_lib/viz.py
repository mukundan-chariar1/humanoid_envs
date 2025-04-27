from configs import constants as _C

from copy import copy
import os; import os.path as osp

import cv2
import imageio
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import pytorch3d
from pytorch3d import transforms
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
)

from pdb import set_trace as st

class Renderer():
    def __init__(self, camera, faces, device):

        self.faces = torch.from_numpy((faces).astype('int')).to(device)
        self.device = device
        self.color = (0.9, 0.9, 0.9)
        self.lights = PointLights(device=device, location=[[-10.0, 0.0, -3.0]])

        if camera is not None:
            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    raster_settings=RasterizationSettings(
                        image_size=(camera.height, camera.width),
                        blur_radius=1e-5, faces_per_pixel=50,)
                ),
                shader=HardPhongShader(device=self.device, lights=self.lights)
            ) 
            self.get_cameras(camera)


    def get_cameras(self, camera, prefix=''):
        setattr(self, f'{prefix}camera', camera)
        setattr(self, f'{prefix}cameras', PerspectiveCameras(
            device=self.device,
            R=transforms.rotation_6d_to_matrix(camera.r6d).transpose(-1, -2), 
            T=camera.T.squeeze(-1),
            focal_length=((_C.CAMERAS.FOCAL_LENGTH, _C.CAMERAS.FOCAL_LENGTH), ),
            principal_point=((_C.CAMERAS.CX, _C.CAMERAS.CY), ),
            image_size=((camera.height, camera.width), ),
            in_ndc=False
        ))

    def render_mesh(self, vertices, prefix='', image=None, alpha=0.8):
        vertices = vertices.unsqueeze(0).to(device=self.device)
        textures = torch.ones_like(vertices)
        textures = textures * torch.tensor(self.color).to(device=self.device)
        mesh = Meshes(verts=vertices, faces=self.faces.unsqueeze(0),
                      textures=TexturesVertex(textures),)
        try: results = self.renderer(mesh, cameras=getattr(self, f'{prefix}cameras'), lights=self.lights)
        except: 
            st()
            return image

        rgb = results[0, ..., :3]
        rgb = torch.flip(rgb, [0, 1]).detach().cpu().numpy()
        depth = results[0, ..., -1]
        depth = torch.flip(depth, [0, 1]).detach().cpu().numpy()

        if image is None: image = np.ones_like(rgb)
        else: image = image.astype(np.float32)/255
        image[depth > 1e-3] = image[depth > 1e-3] * (1-alpha) + (rgb[depth > 1e-3]) * alpha
        return (image * 255).astype(np.uint8)