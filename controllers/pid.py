from configs import constants as _C

import numpy as np
import mujoco
from scipy.linalg import cho_solve, cho_factor

class StablePDController:
    """
    Stable PD (SPD) computes the control forces using the next time step

    .. math::
        \tau^n = -k_p (q^{n+1}-\bar{q}^{n+1}) - k_d \dot{q}^{n+1}

    where :math:`q^n` and :math:`\dot{q}^n` are the position and velocity of the state at time :math:`n`.

    Since :math:`q^{n+1}` and :math:`\dot{q}^{n+1}` are unknown, they are computed using a Taylor expansion

    .. math::
        \tau^n = -k_p (q^{n}+\Delta t \dot{q}^n -\bar{q}^{n+1}) - k_d (\dot{q}^{n+1} + \Delta t \ddot{q}^{n})

    For a nonlinear dynamic systems with multiple degrees of freedo, we compute :math:`\ddot{q}^n` by solving the equation

    .. math::
        M(q) \ddot{q} + C(q,\dot{q}) = \tau_{init} + \tau_{ext}

    where :math:`q`, :math:`\dot{q}` and :math:`\ddot{q}` are vectors of positions, velocities and accelerations of the degrees of freedom, respectively.
    :math:`M(q)` is the mass matrix and :math:`C(q,\dot{q})` is the centrifugal force. :math:`\tau_{ext}` indicates other external force such as gravity.

    See https://www.jie-tan.net/project/spd.pdf for more precise description.


    Note that the desired position is computed from a normalized :math:`action \in [-1, 1]^d`, i.e.,

    .. math::
        \bar{q}^{n+1} = action * action\_scale + action\_offset

    The output is clipped in the range :math:`[torque\_lim, torque\_lim]`.


    Attributes
    ----------
    pd_action_scale : np.ndarray
        scale for action normalization
    pd_action_offset : np.ndarray
        offset for action normalization
    qvel_lim : np.ndarray
        used to extract the matrix :math:`M(q)`
    torque_lim : np.ndarray
        the output is clip in [-torque_lim, torque_lim]
    jkp : np.ndarray
        the :math:`k_p` parameters
    jkd : np.ndarray
        the :math:`k_d` parameters

    """

    def __init__(
        self,
        pd_action_scale: np.ndarray,
        pd_action_offset: np.ndarray,
        qvel_lim: np.ndarray,
        torque_lim: np.ndarray,
        jkp: np.ndarray,
        jkd: np.ndarray,
    ) -> None:
        self.pd_action_scale = pd_action_scale
        self.pd_action_offset = pd_action_offset
        self.qvel_lim = qvel_lim
        self.torque_lim = torque_lim
        self.jkp = jkp
        self.jkd = jkd

    def control(
        self, action: np.ndarray, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> np.ndarray:
        """Computes the clipped torque :math:`\tau^n`.

        Parameters
        ----------
        action : np.ndarray
            action in [-1,1]
        mj_model : mujoco.MjModel
            The mujoco model
        mj_data : mujoco.MjData
            The mujoco data

        Returns
        -------
        np.ndarray
            the torque to be applied
        """
        # scale ctrl to qpos.range

        target_pos = action * self.pd_action_scale + self.pd_action_offset
        
        torque = self._compute_torque(target_pos, mj_model, mj_data)
        torque = np.clip(torque, -self.torque_lim, self.torque_lim)
        return torque

    def _compute_torque(
        self, setpoint: np.ndarray, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> np.ndarray:
        qpos = mj_data.qpos.copy()
        qvel = mj_data.qvel.copy()
        dt = mj_model.opt.timestep
        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        curr_jkp = self.jkp
        curr_jkd = self.jkd
        k_p[6:] = curr_jkp
        k_d[6:] = curr_jkd
        
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:] * dt - setpoint))
        
        qvel_err = qvel
        q_accel = self._compute_desired_accel(
            qpos_err, qvel_err, k_p, k_d, mj_model, mj_data
        )
        qvel_err += q_accel * dt
        torque = -curr_jkp * qpos_err[6:] - curr_jkd * qvel_err[6:]
        return torque

    def _compute_desired_accel(
        self,
        qpos_err: np.ndarray,
        qvel_err: np.ndarray,
        k_p: np.ndarray,
        k_d: np.ndarray,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
    ) -> np.ndarray:
        dt = mj_model.opt.timestep
        nv = mj_model.nv

        M = np.zeros((nv, nv))
        mujoco.mj_fullM(mj_model, M, mj_data.qM)
        M.resize(nv, nv)
        M = M[: self.qvel_lim, : self.qvel_lim]
        C = mj_data.qfrc_bias.copy()[: self.qvel_lim]
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(
            cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]),
            overwrite_b=True,
            check_finite=False,
        )
        return q_accel.squeeze()

def setup_controller(env, clip_actions=True):
    pd_action_scale, pd_action_offset, torque_lim, jkp, jkd=build_pd_action_scale(env, clip_actions)
    controller = StablePDController(pd_action_scale, pd_action_offset, env.qvel_lim, torque_lim, jkp, jkd)

    return controller
            
def build_pd_action_scale(env, clip_actions=True):
    lim_high = np.zeros(env.dof_size)
    lim_low = np.zeros(env.dof_size)
    jkp = np.zeros(env.dof_size)
    jkd = np.zeros(env.dof_size)
    torque_lim = np.zeros(env.dof_size)
    for idx, n in enumerate(env.joint_names[1:]):
        joint_config = env.model.joint(n)
            
        low, high = joint_config.range
        curr_low = low
        curr_high = high
        curr_low = np.max(np.abs(curr_low))
        curr_high = np.max(np.abs(curr_high))
        curr_scale = max([curr_low, curr_high])
        curr_scale = 1.2 * curr_scale
        curr_scale = min([curr_scale, np.pi])

        lim_low[idx] = -curr_scale
        lim_high[idx] = curr_scale
        
    
    for idx, n in enumerate(env.joint_names[1:]):
        joint = "_".join(n.split("_")[:-1])
        jkp[idx] = _C.CONTROL.GAINS[joint][0]
        jkd[idx] = _C.CONTROL.GAINS[joint][1]
        torque_lim[idx] = _C.CONTROL.GAINS[joint][3]
    # self.jkp = self.jkp * 2
    # self.jkd = self.jkd / 5
    
    if clip_actions:
        pd_action_scale = 0.5 * (lim_high - lim_low)
        pd_action_offset = 0.5 * (lim_high + lim_low)
    else:
        pd_action_scale = np.ones_like(lim_high)
        pd_action_offset = np.zeros_like(lim_high)

    return pd_action_scale, pd_action_offset, torque_lim, jkp, jkd