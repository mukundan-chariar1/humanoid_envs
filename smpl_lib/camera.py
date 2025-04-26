from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs import constants as _C

import cv2
import imageio
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from pytorch3d import transforms
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
)

from pdb import set_trace as st


def get_bbox_from_keypoints(keypoints):
    scale = 1.5

    if isinstance(keypoints, torch.Tensor): 
        keypoints = keypoints.detach().cpu().numpy()
    
    keypoints = keypoints[keypoints[:, -1] > 0]
    if keypoints.shape[0] < 2:
        return np.zeros(4)
    
    left, top = keypoints.min(0)[:2]
    right, bottom = keypoints.max(0)[:2]

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    left = max(0, int(cx - width * scale / 2))
    right = min(848, int(cx + width * scale / 2))
    top = max(0, int(cy - height * scale / 2))
    bottom = min(480, int(cy + height * scale / 2))

    return np.array((left, top, right, bottom))


class PerspectiveCamera(nn.Module):
    def __init__(self, faces, device, dtype, is_sideview=False, is_multi_side=False):
        super(PerspectiveCamera, self).__init__()

        self.device = device
        self.dtype = dtype
        self.width, self.height = (640, 480)
        self.is_sideview = is_sideview
        
        K = torch.tensor(
            [[_C.CAMERAS.FOCAL_LENGTH, 0, _C.CAMERAS.CX],
            [0, _C.CAMERAS.FOCAL_LENGTH, _C.CAMERAS.CY],
            [0, 0, 1]]
        ).unsqueeze(0).to(device=device, dtype=dtype)

        if is_sideview:
            r6d = torch.Tensor([[0.3, 0.3, -0.9, -0.4, 0.9, 0.2]]).to(device=device, dtype=dtype)
            T = torch.Tensor([[1.2, -.4, 1.3]]).unsqueeze(-1).to(device=device, dtype=dtype)
            if is_multi_side:
                R=transforms.rotation_6d_to_matrix(r6d)@transforms.axis_angle_to_matrix(torch.tensor([0, torch.pi, 0]).to(device=device, dtype=dtype))
                r6d = transforms.matrix_to_rotation_6d(R.clone())
        else:
            # R = transforms.axis_angle_to_matrix(torch.zeros((1, 3)).to(device=device, dtype=dtype))
            R = transforms.axis_angle_to_matrix(torch.tensor([[0, 0, torch.pi]]).to(device=device, dtype=dtype))@transforms.axis_angle_to_matrix(torch.tensor([[0, torch.pi, 0]]).to(device=device, dtype=dtype))
            r6d = transforms.matrix_to_rotation_6d(R.clone())
            T = torch.zeros((1, 3, 1)).to(device=device, dtype=dtype)
            T[:, 2]=3

        faces = torch.from_numpy(faces.astype('int32')).unsqueeze(0)

        self.register_buffer('K', K.to(dtype=dtype, device=device))
        self.register_buffer('T', T.to(dtype=dtype, device=device))
        self.register_buffer('r6d', r6d.to(dtype=dtype, device=device))
        self.register_buffer('faces', faces.to(device=device))


    def reset_params(self, params):
        for key in params:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = params[key].to(
                    dtype=weight_tensor.dtype, device=weight_tensor.device)
                setattr(self, key, weight_tensor)


    def requires_grad_(self, opt):
        self.r6d.requires_grad_(opt)
        self.T.requires_grad_(opt)


    def update_bbox(self, x3d, scale=1.5):
        """ Update bbox of cameras from the given 3d points
        
        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        x2d = self.project_points_to_image_planes(x3d).squeeze()

        left = torch.clamp(x2d.min(1)[0][:, 0], min=0, max=self.width)
        right = torch.clamp(x2d.max(1)[0][:, 0], min=0, max=self.width)
        top = torch.clamp(x2d.min(1)[0][:, 1], min=0, max=self.height)
        bottom = torch.clamp(x2d.max(1)[0][:, 1], min=0, max=self.height)

        cx = (left + right) / 2
        cy = (top + bottom) / 2
        width = (right - left)
        height = (bottom - top)

        new_left = torch.clamp(cx - width/2 * scale, min=0, max=self.width-1)
        new_right = torch.clamp(cx + width/2 * scale, min=1, max=self.width)
        new_top = torch.clamp(cy - height / 2 * scale, min=0, max=self.height-1)
        new_bottom = torch.clamp(cy + height / 2 * scale, min=1, max=self.height)
        
        bboxes = torch.stack((new_left.detach(), new_top.detach(), 
                              new_right.detach(), new_bottom.detach())).int().float().T
        self.bboxes = bboxes
        
        self.update_intrinsics(bboxes)


    def update_intrinsics(self, bbox):
        K = torch.zeros((4, 4)).to(device=self.device, dtype=self.dtype)
        K[:3, :3] = self.K.clone()
        K[2, 2] = 0
        K[2, -1] = 1
        K[-1, 2] = 1

        left, upper, right, lower = bbox
        cx, cy = K[0, 2], K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[0, 2] = new_cx
        K[1, 2] = new_cy
        self.image_sizes = (int(new_height), int(new_width), )

        if hasattr(self, 'K_full'): self.K_full = K.clone().detach().unsqueeze(0)
        else: self.register_buffer('K_full', K.clone().detach().unsqueeze(0))
        
        self.create_camera()            # update camera
        self.create_renderers()

        if hasattr(self, 'silhouette_renderer'):
            self.update_image_size()    # update renderer

    # Rendering
    def create_camera(self):
        R = transforms.rotation_6d_to_matrix(self.r6d).transpose(-1, -2)
        self.cameras = PerspectiveCameras(
            device=self.device,
            R=R,
            T=self.T.squeeze(-1),
            K=self.K_full,
            image_size=(self.image_sizes, ),
            in_ndc=False)

    # Rendering
    def create_renderers(self, ):
        self.silhouette_renderer, self.mesh_renderer = [], []
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 5.0]])

        ras_silhouette = RasterizationSettings(
            image_size=self.image_sizes,
            blur_radius=0,
            faces_per_pixel=50,
        )

        self.silhouette_renderer.append(MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=ras_silhouette),
            shader=SoftSilhouetteShader(),
        ))

        self.mesh_renderer.append(MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes,
                    blur_radius=1e-5),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
                # cameras=self.cameras
            )
        ))


    def update_image_size(self, ):
        self.silhouette_renderer[0].rasterizer.raster_settings.image_size = self.image_sizes
        self.mesh_renderer[0].rasterizer.raster_settings.image_size = self.image_sizes


    def project_points_to_image_planes(self, x3d):
        """Project 3D points to the cameras coordinate frame
        
        Args:
            x3d: input 3D points (torch.Tensor, [n_f, n_j, 3])
        
        Return:
            x2d: projected 2D points (torch.Tensor, [n_f, n_c, n_j, 2])
        """

        R = transforms.rotation_6d_to_matrix(self.r6d)
        x3d_loc = (R @ x3d.transpose(1, 2) + self.T).transpose(1, 2)
        x3d_hom = torch.div(x3d_loc, x3d_loc[..., 2:])
        x2d = torch.matmul(self.K, x3d_hom.transpose(1, 2)).transpose(1, 2)

        return x2d[..., :2]


    # Rendering
    def overlay_image_onto_background(self, image, mask, background, cam_idx):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        out_image = background.copy()
        bbox = self.bboxes.clone().detach()[cam_idx].int()
        bbox = bbox.cpu().numpy()
        roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        roi_image[mask] = image[mask]
        out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image

        return out_image

    # Rendering
    def render_mesh_to_image_planes(self, vertices):

        mesh = Meshes(verts=vertices, faces=self.faces,
                      textures=TexturesVertex(verts_features=torch.ones_like(vertices) * 0.9),)
        
        self.create_camera()
        results = self.silhouette_renderer[0](
            mesh, cameras=self.cameras[0], lights=self.lights)
        
        results = torch.flip(results, [1, 2])
        return results



def build_cameras(smpl, is_sideview=False, is_multi_side=False):
    return PerspectiveCamera(faces=smpl.model.faces,
                             device=smpl.device,
                             dtype=torch.float32,
                             is_sideview=is_sideview,
                             is_multi_side=is_multi_side)