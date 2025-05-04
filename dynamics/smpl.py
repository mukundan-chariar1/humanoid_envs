import sympy as sp
import mujoco
import numpy as np

from typing import Tuple, List

from utils.symbolic import *
from utils.logger import *

class SMPL:
    def __init__(self):
        self.logger=SimpleFileLogger()
        self.device='cuda'
        self.xml_path = 'models/smpl.xml'
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)

        self.masses=self.model.body_mass[1:]
        self.inertia=self.model.body_inertia[1:]

        self.body_pos=[sp.Matrix(pos) for pos in self.model.body_pos]

        import pdb; pdb.set_trace()



        self.I_i=[sp.diag(m*sp.eye(3), sp.eye(3)*i) for m, i in zip(self.masses, self.inertia)]

        import pdb; pdb.set_trace()



        self.parents = self._build_parent_array()
        self.n_joints = self.model.nbody - 1
        self.joints=[self.model.body(i).name for i in range(self.model.nbody)]

        import pdb; pdb.set_trace()


        self._init_buffers()

        self.q = sp.Matrix([sp.Symbol(f'q{i}') for i in range(75)])
        self.qd = sp.Matrix([sp.Symbol(f'qd{i}') for i in range(75)])

        import pdb; pdb.set_trace()



        self.joint_positions, self.fk_functions=self.init_forward_kinematics(self.q)

        import pdb; pdb.set_trace()


        self.jacobians=self.init_jacobians(self.q)

        import pdb; pdb.set_trace()



        # import pdb; pdb.set_trace()




    def _build_parent_array(self) -> np.array:
        parents = self.model.body_parentid.copy()  # World body has no parent
        parents[0]=-1
        return np.array(parents)
    
    def _init_buffers(self):
        self.local_trans = np.zeros((self.n_joints, 4, 4))
        
        self.world_trans = np.zeros((self.n_joints, 4, 4))

    def init_forward_kinematics(self, 
                                q: sp.Matrix) -> Tuple[List[sp.Matrix], List[sp.Matrix]]:
        n = self.n_joints
        
        world_trans = [None] * n
        joint_positions = [None] * n
        # world_joint_angles = [None] * n

        root_pos = q[:3]
        root_orient = q[3:6]
        joint_angles=q.reshape(25, 3)[2:, :]
        
        # Convert root orientation to rotation matrix
        rot_mat = axis_angle_to_matrix(root_orient)

        # Create root transform matrix (4x4 homogeneous)
        T_root = sp.eye(4)
        T_root[:3, :3] = rot_mat
        T_root[:3, 3] = root_pos
        
        world_trans[0] = T_root
        joint_positions[0] = root_pos
        
        for j in range(1, n):
            parent_idx = self.parents[j]
            
            # Create local transform for current joint
            local_rot = axis_angle_to_matrix(joint_angles[j-1, :])
            T_local = sp.eye(4)
            T_local[:3, :3] = local_rot
            T_local[:3, 3] = self.body_pos[j+1]  # Using stored body positions
            
            # Compute world transform
            world_T = world_trans[parent_idx] * T_local  # Matrix multiplication
            world_trans[j] = world_T
            joint_positions[j] = sp.Matrix(world_T[:3, 3])
            # world_joint_angles=matrix_to_axis_angle(sp.Matrix(world_T[:3, :3]))

        return joint_positions, world_trans
    
    def init_jacobians(self, q: sp.Matrix) -> List[sp.Matrix]:
        """
        Compute symbolic Jacobians of joint positions w.r.t. q.
        
        Returns:
            List of (3 x len(q)) symbolic Jacobians, one per joint.
        """
        jacobians = []

        for trans in self.fk_functions:
            # Compute the Jacobian of the 3D joint position w.r.t. q
            Jbst=[]
            gst_inv=inverse(trans)
            for _q in q:
                gst_dot=sp.diff(trans, _q)
                screw=gst_inv*gst_dot
                Jbst.append(vee(screw))
            jacobians.append(sp.Matrix.hstack(*Jbst))

            import pdb; pdb.set_trace()


        return jacobians

if __name__ == '__main__':
    model = SMPL()
    

    # q = sp.Matrix([sp.Symbol(f'q{i}') for i in range(75)])
    # qd = sp.Matrix([sp.Symbol(f'qd{i}') for i in range(75)])

    # model

    import pdb; pdb.set_trace()



