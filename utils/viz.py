from typing import Dict, Any, Optional, Tuple

import mujoco
import numpy as np
import imageio.v2 as imageio

from envs.base import Env

def setup_renderer(model: mujoco.MjModel, 
                   width: int=640, 
                   height : int=480) -> mujoco.Renderer:
    """Set up offscreen renderer."""
    return mujoco.Renderer(model, width=width, height=height)

def setup_scene_options():
    """Configure visualization options."""
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    return scene_option

def setup_camera(model: mujoco.MjModel,
                 camera_name="track") -> mujoco.MjvCamera:
    """Configure camera settings."""
    camera = mujoco.MjvCamera()
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    
    if cam_id != -1:
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = cam_id
    else:
        print(f"Camera '{camera_name}' not found, using default free camera")
        mujoco.mjv_defaultFreeCamera(model, camera)
    
    return camera

def render_env(env: Env,
               act: list,
               camera_name: str='track',
               output_path: str='output.mp4',
               fps: int=60,
               qpos: Optional[np.array] = None, 
               qvel: Optional[np.array] = None) -> None:
    model=env.model
    data=env.data

    renderer=setup_renderer(model)
    scene_option = setup_scene_options()
    camera = setup_camera(model, camera_name)

    _=env.reset(qpos=qpos, qvel=qvel)

    frame_skip = int(1/(env.dt*fps))

    with imageio.get_writer(output_path, fps=fps) as writer:
        for i, action in enumerate(act):
            env.step(action)

            if i%frame_skip==0:
                renderer.update_scene(env.data, camera=camera, scene_option=scene_option)
                frame = renderer.render()
                writer.append_data(frame)

    renderer.close()
    print(f"Saved video to {output_path}")

def render_image(env: Env,
               act: list,
               camera_name: str='track',
               output_path: str='output.mp4',
               fps: int=60,
               qpos: Optional[np.array] = None, 
               qvel: Optional[np.array] = None) -> None:
    model=env.model
    data=env.data

    renderer=setup_renderer(model)
    scene_option = setup_scene_options()
    camera = setup_camera(model, camera_name)

    _=env.reset(qpos=qpos, qvel=qvel)

    renderer.update_scene(env.data, camera=camera, scene_option=scene_option)
    frame = renderer.render()

    return frame

class Renderer():
    def __init__(self,
                 env: Env,
                 camera_name: str='track',
                 fps: int=60) -> None:
        self.env=env
        self.model=env.model
        self.data=env.data

        self.renderer=setup_renderer(self.model)
        self.scene_option = setup_scene_options()
        self.camera = setup_camera(self.model, camera_name)

        self.frame_skip = int(1/(env.dt*fps))
        self.fps=fps

    def render_env(self,
                   act: list,
                   output_path: str='output.mp4',
                   qpos: Optional[np.array] = None, 
                   qvel: Optional[np.array] = None) -> None:
        _=self.env.reset(qpos=qpos, qvel=qvel)

        with imageio.get_writer(output_path, fps=self.fps) as writer:
            for i, action in enumerate(act):
                self.env.step(action)

                if i%self.frame_skip==0:
                    self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
                    frame = self.renderer.render()
                    writer.append_data(frame)

        print(f"Saved video to {output_path}")

    def render_image(self,
                     qpos: Optional[np.array] = None, 
                     qvel: Optional[np.array] = None) -> np.array:

        _=self.env.reset(qpos=qpos, qvel=qvel)

        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        frame = self.renderer.render()

        return frame
    
    def close(self) -> None:
        self.renderer.close()