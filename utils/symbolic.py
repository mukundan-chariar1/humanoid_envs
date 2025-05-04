import sympy as sp

def axis_angle_to_matrix(axis_angle):
    """
    Convert a symbolic axis-angle vector to a rotation matrix using Rodrigues' formula.

    Args:
        axis_angle: A 3-element sympy Matrix representing the axis-angle vector.

    Returns:
        A 3x3 sympy Matrix representing the rotation matrix.
    """
    x, y, z = axis_angle
    angle = sp.sqrt(x**2 + y**2 + z**2)
    
    # # Avoid division by zero (symbolic safeguard)
    # if angle == 0:
    #     return sp.Matrix.eye(3)

    # Unit axis vector
    # u = axis_angle / angle
    ux=x/angle
    uy=y/angle
    uz=z/angle

    # Skew-symmetric matrix of u
    K = sp.Matrix([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])

    # Rodrigues' rotation formula
    R = sp.Matrix.eye(3) + sp.sin(angle) * K + (1 - sp.cos(angle)) * (K * K)

    return sp.simplify(R)

def matrix_to_axis_angle(matrix: sp.Matrix) -> sp.Matrix:
    """
    Convert a 3x3 symbolic rotation matrix to axis-angle vector.
    The result is a 3D vector whose direction is the rotation axis
    and whose magnitude is the rotation angle (in radians).
    """
    assert matrix.shape == (3, 3), "Rotation matrix must be 3x3"

    # Compute skew-symmetric part for omega
    omega = sp.Matrix([
        matrix[2,1] - matrix[1,2],
        matrix[0,2] - matrix[2,0],
        matrix[1,0] - matrix[0,1]
    ])

    norm_omega = omega.norm()
    trace = matrix.trace()
    theta = sp.atan2(norm_omega, trace - 1)

    # Avoid division by zero in case theta == 0
    axis_angle = sp.Piecewise(
        (sp.Matrix([0, 0, 0]), sp.Eq(theta, 0)),
        (0.5 * omega / sp.sinc(theta / sp.pi), True)
    )

    return theta * axis_angle

def inverse(matrix: sp.Matrix) -> sp.Matrix:
    inv=sp.eye(4)
    R_t=matrix[:3, :3]
    inv[:3, :3]=R_t
    t=inv[:3, 3]
    inv[:3, 3]=R_t*t

    return inv

def vee(matrix: sp.Matrix) -> sp.Matrix:
    v = matrix[:3, 3]
    w = sp.Matrix([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1]
    ]) * 0.5

    return sp.Matrix.vstack(v, w)