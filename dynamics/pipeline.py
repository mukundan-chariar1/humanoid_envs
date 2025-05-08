from dynamics.base import *
from pytorch3d import transforms as T
import mujoco

class Transform(BaseTransform):
    def __init__(self, chain_position, parent, offset):
        super().__init__()
        self.chain_position=chain_position
        self.parent=parent
        self.offset=offset

    def forward(self, q, rotmats, mode):
        if mode==0:
            return self.step(q, rotmats, mode)
        else:
            return self.inverse(q, rotmats, mode)

    def step(self, q, rotmats, mode):
        transform=torch.eye(4)
        transform[:3, 3]=self.offset

        R=T.axis_angle_to_matrix(q[self.chain_position])
        transform[:3, :3]=R

        rotmats[self.chain_position]=rotmats[self.parent]@transform

        return q, rotmats, mode
    
    def inverse(self, q, rotmats, mode):
        inverse_transform=torch.eye(4)
        inverse_transform[:3, :3]=rotmats[self.chain_position, :3, :3].T
        inverse_transform[:3, 3]=-rotmats[self.chain_position, :3, :3].T@rotmats[self.chain_position, :3, 3]

        rotmats[self.chain_position]=inverse_transform

        return q, rotmats, mode
    
class RootTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.chain_position=1
        self.parent=0

    def forward(self, q, rotmats, mode):
        if mode==0:
            return self.step(q, rotmats, mode)
        else:
            return self.inverse(q, rotmats, mode)

    # def step(self, q, rotmats, mode):
    #     transform=torch.eye(4)
    #     transform[:3, 3]=q[0]

    #     R=T.axis_angle_to_matrix(q[self.chain_position])
    #     transform[:3, :3]=R

    #     rotmats[self.chain_position]=rotmats[self.parent]@transform
    #     return q, rotmats, mode

    def step(self, q, rotmats, mode):
        device = q.device
        batch_shape = q.shape[:-2]

        # Get position and orientation
        pos = q[..., 0, :]  # [..., 3]
        axis_angle = q[..., self.chain_position, :]  # [..., 3]
        R = T.axis_angle_to_matrix(axis_angle)  # [..., 3, 3]

        # Build 4x4 transform matrix in batch
        zeros = torch.zeros(*batch_shape, 1, 3, device=device)
        ones = torch.ones(*batch_shape, 1, 1, device=device)

        # Top 3 rows: rotation and translation
        top = torch.cat([R, pos.unsqueeze(-1)], dim=-1)  # [..., 3, 4]

        # Bottom row: [0, 0, 0, 1]
        bottom = torch.cat([zeros, ones], dim=-1)  # [..., 1, 4]

        transform = torch.cat([top, bottom], dim=-2)  # [..., 4, 4]

        # Update rotmats: out-of-place
        rotmats_new = rotmats.clone()
        rotmats_new[..., self.chain_position, :, :] = rotmats[..., self.parent, :, :] @ transform

        return q, rotmats_new, mode


    
    def inverse(self, q, rotmats, mode):
        inverse_transform=torch.eye(4)
        inverse_transform[:3, :3]=rotmats[self.chain_position, :3, :3].T
        inverse_transform[:3, 3]=-rotmats[self.chain_position, :3, :3].T@rotmats[self.chain_position, :3, 3]

        rotmats[self.chain_position]=inverse_transform

        return q, rotmats, mode
    
class KinematicChain(nn.Module):
    def __init__(self, offsets, parents):
        super().__init__()
        self.offsets=offsets
        self.parents = parents

        self.transforms=[RootTransform()]
        for chain_position, parent, offset in zip(range(2, self.offsets.shape[0]), self.parents[2:], self.offsets[2:]):
            self.transforms.append(Transform(chain_position, parent, offset))

    def forward(self, q, rotmats, mode):
        q=q.reshape(-1, 3)
        for joint in self.transforms:
            _, rotmats, _ = joint(q, rotmats, mode)
        return rotmats
    
    # def derivative(self, q, eps=1e-6):
    #     derivatives=[]

    #     for i in range(q.shape[0]):
    #         q_m2 = q.clone(); q_m2[i] -= 2 * eps
    #         q_m1 = q.clone(); q_m1[i] -= eps
    #         q_p1 = q.clone(); q_p1[i] += eps
    #         q_p2 = q.clone(); q_p2[i] += 2 * eps

    #         rotmats = torch.stack([torch.eye(4)] * 25)

    #         f_m2 = self(q_m2, rotmats.clone(), 0)
    #         f_m1 = self(q_m1, rotmats.clone(), 0)
    #         f_p1 = self(q_p1, rotmats.clone(), 0)
    #         f_p2 = self(q_p2, rotmats.clone(), 0)

    #         deriv = (-f_p2 + 8*f_p1 - 8*f_m1 + f_m2) / (12 * eps)
    #         derivatives.append(deriv)

    #     return torch.stack(derivatives, axis=1)

    def derivative(self, q, eps=1e-6):
        n = q.shape[0]  # number of joints
        device = q.device
        dtype = q.dtype

        # Create identity matrix [n, n] for unit perturbations
        eye = torch.eye(n, device=device, dtype=dtype)

        # Perturbations for 4th order finite differences: [-2, -1, +1, +2] * eps
        shifts = torch.tensor([-2, -1, 1, 2], device=device, dtype=dtype).view(4, 1, 1) * eps  # [4, 1, 1]
        perturbations = shifts * eye.unsqueeze(0)  # [4, n, n]

        # Perturbed q's: [4, n, n]
        q_perturbed = q.unsqueeze(0) + perturbations  # [4, n, n]

        # Create function to evaluate the transform stack
        def eval_fn(qi):
            rotmats = torch.stack([torch.eye(4, device=device)] * 25)
            return self(qi, rotmats, 0)  # shape: [25, 4, 4]

        # Evaluate all 4 perturbations across n joints using nested vmap
        f_vals = torch.vmap(torch.vmap(eval_fn))(q_perturbed)  # shape: [4, n, 25, 4, 4]

        f_m2, f_m1, f_p1, f_p2 = f_vals[0], f_vals[1], f_vals[2], f_vals[3]

        # 4th-order central difference
        derivs = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * eps)  # shape: [n, 25, 4, 4]

        return derivs.permute(1, 2, 3, 0)  # final shape: [25, 4, 4, n]

    def body_jacobian(self, q, rotmats, eps=1e-6):
        inv_rotmat=self(q, rotmats.clone(), 1)
        derivatives=self.derivative(q, eps=eps)
        jacobian_rotmats=inv_rotmat.unsqueeze(1)@derivatives

        body_jacobian=self.vee(jacobian_rotmats)

        return body_jacobian

    def vee(self, jacobian_rotmats):
        v = jacobian_rotmats[:, :, :3, 3].permute(0, 2, 1)

        w=torch.stack([jacobian_rotmats[:, :, 2, 1]-jacobian_rotmats[:, :, 1, 2],
                       jacobian_rotmats[:, :, 0, 2]-jacobian_rotmats[:, :, 2, 0],
                       jacobian_rotmats[:, :, 1, 0]-jacobian_rotmats[:, :, 0, 1]], axis=1)*0.5

        return torch.concatenate([v, w], axis=1)

class DynamicsChain(nn.Module):
    def __init__(self, masses, inertia, offsets, parents):
        super().__init__()
        self.masses=masses
        self.inertia=inertia

        self.kinematics=KinematicChain(offsets, parents)

        self.M_i=torch.stack([torch.block_diag(m*torch.eye(3), i*torch.eye(3)) for m, i in zip(self.masses, self.inertia)]).float()

    def mass_matrix(self, q, rotmats, eps=1e-6):
        jacobians=self.kinematics.body_jacobian(q, rotmats, eps=eps)

        mass_matrix=jacobians.permute(0, 2, 1)@self.M_i@jacobians
        return mass_matrix.sum(0)
    
    def coriolis_matrix(self, q, qd, rotmats, eps=1e-6):
        n=q.shape[0]
        C=torch.zeros((n, n))
        
        dMdq = []
        for k in range(n):
            q_k_p = q.clone(); q_k_p[k] += eps
            q_k_m = q.clone(); q_k_m[k] -= eps

            M_k_p = self.mass_matrix(q_k_p, rotmats, eps)
            M_k_m = self.mass_matrix(q_k_m, rotmats, eps)

            dMdq_k = (M_k_p - M_k_m) / (2 * eps)
            dMdq.append(dMdq_k)

        dMdq=torch.stack(dMdq)

        # for i in range(n):
        #     for j in range(n):
        #         for k in range(n):
        #             term = (
        #                 dMdq[k, i, j] + dMdq[j, i, k] - dMdq[i, j, k]
        #             ) * qd[k] / 2.0
        #             C[i, j] += term

        dMdq_ijk = dMdq.permute(1, 2, 0)
        dMdq_jik = dMdq.permute(2, 1, 0)
        dMdq_ijk2 = dMdq

        christoffel_terms = 0.5 * (
                                dMdq_ijk + dMdq_jik - dMdq_ijk2
                            )
        C = torch.einsum('ijk,k->ij', christoffel_terms, qd)
        return C

if __name__=='__main__':
    xml_path = 'models/smpl.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)

    parents = torch.tensor(model.body_parentid.copy())
    parents[0]=-1

    offsets=torch.tensor(model.body_pos.copy())

    masses=torch.tensor(model.body_mass)
    inertia=torch.tensor(model.body_inertia)

    model=DynamicsChain(masses, inertia, offsets, parents)

    # q=torch.zeros(75)
    # q[2]=0.9205
    q=torch.rand(75)
    # qd=torch.rand(75)

    rotmats=torch.stack([torch.eye(4)]*25)
    derivatives=model.kinematics.derivative(q)

    import pdb; pdb.set_trace()

