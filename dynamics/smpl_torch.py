import torch
import mujoco
from dynamics.base import BaseDynamics

from pytorch3d.transforms import *

from typing import Tuple

class SMPL(BaseDynamics):
    def __init__(self):
        self.device='cuda'
        self.xml_path = 'models/smpl.xml'
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)

        self.masses=self.model.body_mass[1:]
        self.inertia=torch.tensor(self.model.body_inertia[1:])

        self.M_i=torch.stack([torch.block_diag(m*torch.eye(3), i*torch.eye(3)) for m, i in zip(self.masses, self.inertia)]).to(torch.float64).to(self.device)

        self.parents = self._build_parent_array()
        self.n_joints = self.model.nbody - 1
        self._init_buffers()

    def _build_parent_array(self) -> torch.Tensor:
        parents = self.model.body_parentid.copy()  # World body has no parent
        parents[0]=-1
        return torch.tensor(parents, device=self.device, dtype=torch.long)
    
    def _init_buffers(self):
        self.local_trans = torch.zeros((self.n_joints, 4, 4), 
                                      device=self.device)
        
        self.world_trans = torch.zeros((self.n_joints, 4, 4), 
                                       device=self.device)

        self.joint_positions = torch.from_numpy(self.model.body_pos[1:]).to(self.device)

    def forward_kinematics(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._forward_kinematics(q[:3], quaternion_to_axis_angle(q[3:7].unsqueeze(0))[0], q[7:].reshape((-1, 3)))
        
    def _forward_kinematics(self, 
                            root_pos: torch.Tensor, 
                            root_orient: torch.Tensor, 
                            joint_angles: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.n_joints

        world_trans = [None] * n
        joint_positions = [None] * n

        rot_mat = axis_angle_to_matrix(root_orient)
        T_root = torch.eye(4, device=self.device)
        T_root[:3, :3] = rot_mat
        T_root[:3, 3] = root_pos

        world_trans[0] = T_root
        joint_positions[0] = root_pos

        for j in range(1, n):
            parent_idx = self.parents[j]

            local_rot = axis_angle_to_matrix(joint_angles[j - 1])
            T_local = torch.eye(4, device=self.device)
            T_local[:3, :3] = local_rot

            offset = torch.tensor(self.model.body_pos[j+1], device=self.device)
            T_local[:3, 3] = offset

            world_T = world_trans[parent_idx] @ T_local
            world_trans[j] = world_T
            joint_positions[j] = world_T[:3, 3]

        return torch.stack(joint_positions).to(torch.float64), torch.stack(world_trans).to(torch.float64)
    
    def _process_root(self, 
                  root_pos: torch.Tensor, 
                  root_orient: torch.Tensor):
        rot_mat = axis_angle_to_matrix(root_orient)
        trans_mat = torch.eye(4, device=self.device)
        trans_mat = trans_mat.clone()
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = root_pos

        self.world_trans = self.world_trans.clone()
        self.local_trans = self.local_trans.clone()
        self.joint_positions = self.joint_positions.clone()

        self.world_trans[0] = trans_mat.clone()
        self.local_trans[0] = trans_mat.clone()
        self.joint_positions[0] = root_pos.clone()

    def _process_joint(self, 
                    joint_idx: int, 
                    joint_angle: torch.Tensor):
        parent_idx = self.parents[joint_idx]

        local_rot = axis_angle_to_matrix(joint_angle)
        local_trans = torch.eye(4, device=self.device)
        local_trans = local_trans.clone()
        local_trans[:3, :3] = local_rot

        body = self.model.body(joint_idx + 1)
        offset = torch.tensor(body.pos, device=self.device, dtype=torch.float32)
        local_trans[:3, 3] = offset

        self.local_trans[joint_idx] = local_trans.clone()

        world_T = self.world_trans[parent_idx] @ local_trans
        self.world_trans[joint_idx] = world_T.clone()
        self.joint_positions[joint_idx] = world_T[:3, 3].clone()


    def compute_jacobians(self,
                          root_pos: torch.Tensor,
                          root_orient: torch.Tensor,
                          joint_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_angles = joint_angles.clone().detach().requires_grad_(True)
        root_pos = root_pos.clone().detach().requires_grad_(True)
        root_orient = root_orient.clone().detach().requires_grad_(True)

        joint_pos, world_trans = self._forward_kinematics(root_pos, root_orient, joint_angles)
        n_dof = joint_angles.numel() + 6

        J_pos = torch.zeros((self.n_joints, 3, n_dof), dtype=joint_angles.dtype, device=self.device)
        J_rot = torch.zeros((self.n_joints, 3, n_dof), dtype=joint_angles.dtype, device=self.device)

        for i in range(self.n_joints):
            for d in range(3):
                grad_outputs = torch.zeros_like(joint_pos)
                grad_outputs[i, d] = 1.0
                grad = torch.autograd.grad(joint_pos, [root_pos, root_orient, joint_angles],
                                        grad_outputs=grad_outputs, retain_graph=True, create_graph=True)
                J_pos[i, d, :3] = grad[0]
                J_pos[i, d, 3:6] = grad[1][:3]
                J_pos[i, d, 6:] = grad[2].reshape(-1)

            for d in range(3):
                rot = matrix_to_axis_angle(world_trans[i, :3, :3].unsqueeze(0)).squeeze(0)
                grad_outputs = torch.zeros_like(rot)
                grad_outputs[d] = 1.0
                grad = torch.autograd.grad(rot, [root_pos, root_orient, joint_angles],
                                        grad_outputs=grad_outputs, retain_graph=True, create_graph=True)
                J_rot[i, d, :3] = grad[0]
                J_rot[i, d, 3:6] = grad[1][:3]
                J_rot[i, d, 6:] = grad[2].reshape(-1)

        return torch.cat((J_pos, J_rot), dim=1)
    
    def compute_jacobians_vector(self,
                        root_pos: torch.Tensor,
                        root_orient: torch.Tensor,
                        joint_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine inputs for unified autograd
        q = torch.cat([root_pos, root_orient, joint_angles.view(-1)])
        q = q.detach().requires_grad_(True)
        n_dof = q.shape[0]

        def fk_joint_pos(q_input):
            root_pos = q_input[:3]
            root_orient = q_input[3:6]
            joint_angles = q_input[6:]
            joint_pos, _ = self._forward_kinematics(root_pos, root_orient, joint_angles)
            return joint_pos.view(-1)  # (n_joints * 3)

        def fk_joint_rot(q_input):
            root_pos = q_input[:3]
            root_orient = q_input[3:6]
            joint_angles = q_input[6:]
            _, world_trans = self._forward_kinematics(root_pos, root_orient, joint_angles)
            rot_mats = world_trans[:, :3, :3]  # (n_joints, 3, 3)
            axis_angles = matrix_to_axis_angle(rot_mats)  # (n_joints, 3)
            return axis_angles.view(-1)  # (n_joints * 3)

        # Compute Jacobians (output_dim, input_dim)
        J_pos_full = torch.autograd.functional.jacobian(fk_joint_pos, q, create_graph=True)
        J_rot_full = torch.autograd.functional.jacobian(fk_joint_rot, q, create_graph=True)

        # Reshape to (n_joints, 3, n_dof)
        J_pos = J_pos_full.view(self.n_joints, 3, n_dof)
        J_rot = J_rot_full.view(self.n_joints, 3, n_dof)

        return torch.cat((J_pos, J_rot), dim=1)  # (n_joints, 6, n_dof)

    
    def compute_mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        return self._compute_mass_matrix(q[:3], quaternion_to_axis_angle(q[3:7].unsqueeze(0))[0], q[7:].reshape((-1, 3)))
    
    def _compute_mass_matrix(self,
                        root_pos: torch.Tensor,
                        root_orient: torch.Tensor,
                        joint_angles: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            joint_pos, world_trans = self._forward_kinematics(root_pos, root_orient, joint_angles)

            X_world_to_body = [self.transform_to_spatial(world_T.inverse()) for world_T in world_trans]

            I_composite = list(self.M_i)

            for i in reversed(range(1, self.n_joints)):
                parent = self.parents[i]
                X_pi = self.spatial_transform_between(world_trans[parent], world_trans[i])
                I_composite[parent] = I_composite[parent] + self.transform_spatial_inertia(X_pi, I_composite[i])

            n_dof = 6 + 3 * (self.n_joints - 1)
            M = torch.zeros((n_dof, n_dof), dtype=torch.float32, device=self.device)

            S = [torch.eye(6, device=self.device, dtype=torch.float64) if i == 0 else self.joint_motion_subspace(i) for i in range(self.n_joints)]

            for i in range(self.n_joints):
                for j in range(i + 1):
                    F = I_composite[i] @ S[i]
                    M_ij = S[j].T @ F
                    idx_i = self._dof_index(i)
                    idx_j = self._dof_index(j)
                    M[idx_i:idx_i+3, idx_j:idx_j+3] = M_ij[:3, :3]

                    if i != j:
                        M[idx_j:idx_j+3, idx_i:idx_i+3] = M_ij[:3, :3].T

            M = 0.5 * (M + M.T)

            return M
        
    def spatial_transform_between(self, T_parent: torch.Tensor, T_child: torch.Tensor) -> torch.Tensor:
        T_pc = torch.inverse(T_parent) @ T_child
        X_pc = self.transform_to_spatial(T_pc)

        return X_pc

    def transform_spatial_inertia(self, X: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        return X.T @ I @ X
    
    def transform_to_spatial(self, T: torch.Tensor) -> torch.Tensor:
        R = T[:3, :3]
        p = T[:3, 3]

        X = torch.zeros((6, 6), device=self.device, dtype=torch.double)
        X[:3, :3] = R
        X[3:, :3] = self.skew(p) @ R
        X[3:, 3:] = R
        return X

    def skew(self, v: torch.Tensor) -> torch.Tensor:
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], device=self.device)

    def joint_motion_subspace(self, joint_idx: int) -> torch.Tensor:
        return torch.eye(6, device=self.device, dtype=torch.float64)[:, :3]

    def _dof_index(self, joint_idx: int) -> int:
        if joint_idx == 0:
            return 0
        return 6 + 3 * (joint_idx - 1)

    def compute_coriolis_matrix(self, root_pos, root_orient, joint_angles, q_dot):
        """
        Compute Coriolis matrix C(q, q_dot) using Jacobian of M(q).
        
        Args:
            mass_matrix_func: Function taking (q,) and returning (n, n) mass matrix.
            q: Tensor of shape (n,) - joint positions.
            q_dot: Tensor of shape (n,) - joint velocities.
            
        Returns:
            C: Coriolis matrix of shape (n, n)
        """
        n = q_dot.shape[0]
        
        # Compute dM/dq_k → shape: (n, n, n)
        # dM_dq[i,j,k] = ∂M_ij / ∂q_k
        dM_dq = torch.autograd.functional.jacobian(self._compute_mass_matrix, root_pos)  # shape (n, n, n)

        import pdb; pdb.set_trace()



        # Compute Coriolis matrix C_ij = sum_k c_ijk * q_dot_k
        C = torch.zeros(n, n, dtype=q.dtype)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    c_ijk = 0.5 * (dM_dq[i, j, k] + dM_dq[i, k, j] - dM_dq[j, k, i])
                    C[i, j] += c_ijk * q_dot[k]
        
        return C



if __name__ == '__main__':
    model = SMPL()

    # torch.autograd.set_detect_anomaly(True)

    # Example input: root translation, orientation (quaternion), and joint angles (axis-angle)
    root_pos = torch.tensor([0.0, 0.9, 0.0], dtype=torch.float32).to(model.device)
    root_orient = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).to(model.device)  # identity quaternion
    joint_angles = torch.zeros((model.n_joints - 1, 3), dtype=torch.float32).to(model.device)  # zero pose
    joint_velocities = torch.zeros((model.n_joints+1, 3), dtype=torch.float32).to(model.device)

    root_pos.requires_grad_(True)
    root_orient.requires_grad_(True)
    joint_angles.requires_grad_(True)
    joint_velocities.requires_grad_(True)

    # Enable gradient tracking for Jacobian computation
    root_pos.requires_grad_(True)
    root_orient.requires_grad_(True)
    joint_angles.requires_grad_(True)

    M=model.compute_mass_matrix(torch.cat([root_pos, root_orient, joint_angles.view(-1)]))

    import pdb; pdb.set_trace()


    C=model.compute_coriolis_matrix(root_pos, root_orient[1:], joint_angles.view(-1), joint_velocities)

    # import pdb; pdb.set_trace()


