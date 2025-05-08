import numpy as np
import cv2

from envs.smpl import *

from utils.transforms import *
from utils.utils import *
from utils.load_traj import get_traj_from_wham

from utils.viz import Renderer

def stage_cost(Q, R, xref, uref, x, u, k):
    return 0.5 * ((convert_to_x_aa(x) - convert_to_x_aa(xref[k])).T @ Q @ (convert_to_x_aa(x) - convert_to_x_aa(xref[k])) + (u - uref[k]).T @ R @ (u - uref[k]))


def term_cost(Qf, xref, x):
    return 0.5 * (convert_to_x_aa(x) - convert_to_x_aa(xref[-1])).T @ Qf @ (convert_to_x_aa(x) - convert_to_x_aa(xref[-1]))


def stage_cost_expansion(Q, R, xref, uref, x, u, k):
    grad_xx = Q
    grad_x = Q @ (convert_to_x_aa(x) - convert_to_x_aa(xref[k]))
    grad_uu = R
    grad_u = R @ (u - uref[k])
    return grad_xx, grad_x, grad_uu, grad_u


def term_cost_expansion(Qf, xref, x):
    grad_xx = Qf
    grad_x = Qf @ (convert_to_x_aa(x) - convert_to_x_aa(xref[-1]))
    return grad_xx, grad_x


def backward_pass(Q, R, Qf, xref, uref, X, U, env):
    N=X.shape[0]
    nx=env.model.nv*2
    nu=uref.shape[-1]

    P = np.zeros((N, nx, nx))
    p = np.zeros((N, nx))
    d = np.zeros((N-1, nu))
    K = np.zeros((N-1, nu, nx))

    Delta_J = 0.0

    grad_xx_terminal, grad_x_terminal = term_cost_expansion(Qf, xref, X[-1])
    P[-1] = grad_xx_terminal
    p[-1] = grad_x_terminal

    for i in reversed(range(N-1)):
        A, B=linearize_dynamics_for_control(env, X[i], U[i])

        grad_xx, grad_x, grad_uu, grad_u = stage_cost_expansion(Q, R, xref, uref, X[i], U[i], i)

        G_x = grad_x + A.T @ p[i+1]
        G_u = grad_u + B.T @ p[i+1]
        G_xx = grad_xx + A.T @ P[i+1] @ A
        G_uu = grad_uu + B.T @ P[i+1] @ B
        G_ux = B.T @ P[i+1] @ A
        G_xu = A.T @ P[i+1] @ B

        G_uu_reg = grad_uu + B.T @ (P[i+1]+1e-6 * np.eye(P[i+1].shape[1])) @ B
        
        d[i] = np.linalg.solve(G_uu_reg, G_u)
        K[i] = np.linalg.solve(G_uu_reg, G_ux)

        P[i] = G_xx + K[i].T @ G_uu @ K[i] - G_ux.T @ K[i] - K[i].T @ G_ux
        p[i] = G_x - K[i].T @ G_u + K[i].T @ G_uu @ d[i] - G_ux.T @ d[i]

        Delta_J += 0.5 * d[i].T @ G_uu @ d[i] + d[i].T @ G_u

    return d, K, Delta_J


def trajectory_cost(Q, R, Qf, xref, uref, X, U):
    J = 0.0
    N = xref.shape[0]
    for i in range(N-1):
        J += stage_cost(Q, R, xref, uref, X[i], U[i], i)
    J += term_cost(Qf, xref, X[-1])
    return J


def forward_pass(Q, R, Qf, xref, uref, X, U, d, K, env, max_linesearch_iters=20):
    N, nx=xref.shape
    nu=uref.shape[-1]

    Xn = np.zeros((N, nx))
    Un = np.zeros((N-1, nu))

    Xn[0] = X[0].copy()
    alpha = 1.0

    for _ in range(max_linesearch_iters):
        _=env.reset(qpos=np.concatenate([Xn[0, :Xn.shape[1]//2+1]]), qvel=Xn[0, Xn.shape[1]//2+1:])
        for j in range(N-1):
            Un[j] = U[j] - alpha * d[j] - K[j] @ (convert_to_x_aa(Xn[j]) - convert_to_x_aa(X[j]))
            if np.isnan(Un[j]).any() or np.isinf(Un[j]).any(): 

                import pdb; pdb.set_trace()


            obs, _, _, _ = env.step(Un[j])
            Xn[j+1]=np.concatenate([obs['qpos'], obs['qvel']])

        J_new = trajectory_cost(Q, R, Qf, xref, uref, Xn, Un)
        if J_new < trajectory_cost(Q, R, Qf, xref, uref, X, U):
            return Xn, Un, J_new, alpha

        alpha *= 0.5

    raise RuntimeError("forward pass failed")


def iLQR(Q, R, Qf, xref, uref, U, env, atol=1e-3, max_iters=250, verbose=True):
    N, nx=xref.shape
    nu=uref.shape[-1]

    assert len(U) == N-1
    assert len(U[0]) == nu

    X = np.zeros((N, nx))
    X[0] = xref[0].copy()

    _=env.reset(qpos=X[0, :X.shape[1]//2+1], qvel=X[0, X.shape[1]//2+1:])
    for i in range(N-1):
        obs, _, _, _ = env.step(U[i])
        X[i+1] = np.concatenate([obs['qpos'], obs['qvel']])

    J = trajectory_cost(Q, R, Qf, xref, uref, X, U)

    for ilqr_iter in range(max_iters):
        d, K, Delta_J = backward_pass(Q, R, Qf, xref, uref, X, U, env)
        X, U, J, alpha = forward_pass(Q, R, Qf, xref, uref, X, U, d, K, env, max_linesearch_iters=40)

        if Delta_J < atol:
            if verbose:
                print("iLQR converged")
            return X, U, K

        if verbose:
            dmax = max(np.linalg.norm(di) for di in d)
            if ilqr_iter % 10 == 0:
                print(f"iter     J           ΔJ        |d|         α")
                print("-------------------------------------------------")
            print(f"{ilqr_iter:3d}   {J:10.3e}  {Delta_J:9.2e}  {dmax:9.2e}  {alpha:6.4f}")

    raise RuntimeError("iLQR failed")

def piecewise_ilqr(Q, R, Qf, xref, uref, U, env, segment_len=20):
    N = xref.shape[0]
    segments = []
    x0 = xref[0]

    for i in range(0, N-1, segment_len):
        start, end = i, min(i + segment_len, N)
        xref_seg = xref[start:end]
        uref_seg = uref[start:end-1]
        U_seg = U[start:end-1]

        env.reset(qpos=x0[:env.model.nq], qvel=x0[env.model.nq:])
        X_seg, U_seg, K_seg = iLQR(Q, R, Qf, xref_seg, uref_seg, U_seg, env, atol=1e2)

        x0 = X_seg[-1]
        segments.append((X_seg, U_seg, K_seg))

    return segments

def get_piecewise_control(segments, t, x):
    for i, (X, U, K) in enumerate(segments):
        seg_len = len(U)
        if t < seg_len:
            xref = X[t]
            u_ff = U[t]
            K_fb = K[t]
            u = u_ff - K_fb @ (convert_to_x_aa(x) - convert_to_x_aa(xref))
            return u
        t -= seg_len
    return np.zeros_like(U[0])

def receding_horizon_ilqr(Q, R, Qf, xref, uref, U_init, env, horizon=30, total_steps=150, verbose=True):
    nx = xref.shape[1]
    nu = uref.shape[1]

    x0 = xref[0]
    _ = env.reset(qpos=x0[:env.model.nq], qvel=x0[env.model.nq:])
    X = [x0]
    U_applied = []

    for t in range(total_steps - 1):
        remaining = total_steps - t
        H = min(horizon, remaining)

        xref_seg = xref[t:t+H]
        uref_seg = uref[t:t+H-1]
        U_seg = U_init[t:t+H-1].copy()

        # Reset env to current state
        env.reset(qpos=X[-1][:env.model.nq], qvel=X[-1][env.model.nq:])

        # Solve iLQR from current state
        try:
            X_sol, U_sol, _ = iLQR(Q, R, Qf, xref_seg, uref_seg, U_seg, env, atol=5e1, max_iters=100, verbose=verbose)
        except RuntimeError:
            print(f"iLQR failed at time {t}")
            break

        # Apply the first control
        u = U_sol[0]
        obs, _, _, _ = env.step(u)
        x_next = np.concatenate([obs['qpos'], obs['qvel']])

        X.append(x_next)
        U_applied.append(u)

    return np.array(X), np.array(U_applied)


# if __name__=='__main__':
#     env = SMPL(num_simulation_step_per_control_step=5)
#     state = env.reset()
#     renderer = Renderer(env)

#     nv = env.model.nv
#     nu = env.model.nu

#     rot, ang, transl_, vel, root_rot, root_ang = get_traj_from_wham(env.dt)

#     root_rot = axis_angle_to_quaternion(np.zeros_like(root_rot))
#     rot = np.concatenate((root_rot, smpl_to_robot(rot)), axis=-1)
#     ang = np.concatenate((np.zeros_like(root_ang), smpl_to_robot(ang)), axis=-1)

#     transl_[:, 2]=0
#     transl = transl_ + env.initial_pose[:3]

#     q_ref = np.concatenate((transl, rot), axis=-1)
#     qd_ref = np.concatenate((vel, ang), axis=-1)
#     xref = np.concatenate([q_ref, qd_ref], axis=-1)[:300]

#     U = np.ones((xref.shape[0]-1, nu)) * 0.1
#     R = np.eye(nu)

#     jac_com = np.zeros((3, nv))
#     # mujoco.mj_jacSubtreeCom(env.model, env.data, jac_com, env.model.body('Head').id)
#     mujoco.mj_jacSubtreeCom(env.model, env.data, jac_com, env.model.body('Torso').id)
#     jac_rfoot = np.zeros((3, nv))
#     mujoco.mj_jacBodyCom(env.model, env.data, jac_rfoot, None, env.model.body('R_Ankle').id)
#     jac_lfoot = np.zeros((3, nv))
#     mujoco.mj_jacBodyCom(env.model, env.data, jac_lfoot, None, env.model.body('L_Ankle').id)

#     jac_rdiff = jac_com - jac_rfoot
#     jac_ldiff = jac_com - jac_lfoot
#     Qbalance = jac_rdiff.T @ jac_rdiff + jac_ldiff.T @ jac_ldiff

#     Qbalance=np.block([[Qbalance, np.zeros((nv, nv))],
#                   [np.zeros((nv, 2*nv))]])

#     Q = np.block([[np.eye(nv, nv), np.zeros((nv, nv))],
#                   [np.zeros((nv, 2*nv))]])
#     Q[0:3, 0:3] *= 100
#     # Q[3:6, 3:6] *= 0
#     Q[nv:nv+3, nv:nv+3] = 1
#     # Q*=5
#     Q+=Qbalance*1000
#     Qf = Q.copy()

#     X_sol, U_sol = receding_horizon_ilqr(Q, R, Qf, xref, U, U, env, horizon=20, total_steps=len(xref))

#     renderer.render_env(act=np.array(U_sol), qpos=q_ref[0], qvel=qd_ref[0])


# if __name__ == '__main__':
#     env = SMPL(num_simulation_step_per_control_step=5)
#     state = env.reset()
#     renderer = Renderer(env)

#     nv = env.model.nv
#     nu = env.model.nu

#     rot, ang, transl_, vel = get_traj_from_wham(env.dt)
#     root_rot = axis_angle_to_quaternion(np.zeros((rot.shape[0], 3)))
#     rot = np.concatenate((root_rot, smpl_to_robot(rot)), axis=-1)
#     ang = np.concatenate((np.zeros((root_rot.shape[0], 3)), smpl_to_robot(ang)), axis=-1)
#     transl = transl_ + env.initial_pose[:3]

#     q_ref = np.concatenate((transl, rot), axis=-1)
#     qd_ref = np.concatenate((vel, ang), axis=-1)
#     xref = np.concatenate([q_ref, qd_ref], axis=-1)[:150]

#     U = np.ones((xref.shape[0]-1, nu)) * 0.1
#     R = np.eye(nu) * 1e-6
#     Q = np.block([[np.eye(nv, nv), np.zeros((nv, nv))],
#                   [np.zeros((nv, 2*nv))]])
#     Q[0:3, 0:3] *= 10
#     Q[3:6, 3:6] *= 0
#     Q[nv:nv+3, nv:nv+3] = 1
#     Q*=5
#     Qf = Q.copy() * 10

#     segments = piecewise_ilqr(Q, R, Qf, xref, U, U, env, segment_len=30)

#     obs=env.reset(qpos=q_ref[0], qvel=qd_ref[0])
#     act = []
#     for t in range(len(xref)-1):
#         x = np.concatenate([obs['qpos'], obs['qvel']])
#         u = get_piecewise_control(segments, t, x)
#         act.append(u)
#         env.step(u)

#     renderer.render_env(act=np.array(act), qpos=q_ref[0], qvel=qd_ref[0])

if __name__=='__main__':
    env = SMPL(num_simulation_step_per_control_step=5)
    state = env.reset()
    env_renderer=Renderer(env)

    nv=env.model.nv
    nu = env.model.nu

    rot, ang, transl_, vel, root_rot, root_ang = get_traj_from_wham(env.dt)

    root_rot = axis_angle_to_quaternion(np.zeros_like(root_rot))
    rot = np.concatenate((root_rot, smpl_to_robot(rot)), axis=-1)
    ang = np.concatenate((np.zeros_like(root_ang), smpl_to_robot(ang)), axis=-1)

    transl_[:, 2]=0
    transl = transl_ + env.initial_pose[:3]

    q_ref = np.concatenate((transl, rot), axis=-1)
    qd_ref = np.concatenate((vel, ang), axis=-1)
    xref = np.concatenate([q_ref, qd_ref], axis=-1)[:300]


    U=np.ones((xref.shape[0]-1, nu))*0.1

    R = np.eye(nu)*1e-6
    Q = np.block([[np.eye(nv, nv), np.zeros((nv, nv))],
              [np.zeros((nv, 2*nv))]])
    # Q=np.eye(nv*2)
    # Q*=0.01
    Q[3:6, 3:6]*=0
    Q[nv:nv+3, nv:nv+3]=1
    Qf=Q.copy()*10

    X, U, K=iLQR(Q, R, Qf, xref, U, U, env, atol=1e1)

    env_renderer.render_env(act=U, qpos=q_ref[0], qvel=qd_ref[0])

