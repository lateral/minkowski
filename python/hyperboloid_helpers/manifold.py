import numpy as np


def minkowski_dot(u, v):
    """
    `u` and `v` are vectors in Minkowski space.
    """
    rank = u.shape[-1] - 1
    euc_dp = u[:rank].dot(v[:rank])
    return euc_dp - u[rank] * v[rank]


def minkowski_dot_matrix(vecs_a, vecs_b):
    """
    Return the matrix giving the Minkowski dot product of every vector in vecs_a with every vector in vecs_b.
    """
    rank = vecs_a.shape[1] - 1
    euc_dps = vecs_a[:,:rank].dot(vecs_b[:,:rank].T)
    timelike = vecs_a[:,rank][:,np.newaxis].dot(vecs_b[:,rank][:,np.newaxis].T)
    return euc_dps - timelike


def logarithm(base, other):
    """
    Return the logarithm of `other` in the tangent space of `base`.
    """
    mdp = minkowski_dot(base, other)
    dist = np.arccosh(-mdp)
    proj = other + mdp * base
    norm = np.sqrt(minkowski_dot(proj, proj)) 
    proj *= dist / norm
    return proj


def exponential(base, tangent):
    """
    Compute the exponential of `tangent` from the point `base`.
    """
    tangent = tangent.copy()
    norm = np.sqrt(minkowski_dot(tangent, tangent))
    tangent /= norm
    return np.cosh(norm) * base + np.sinh(norm) * tangent


def geodesic_parallel_transport(base, direction, tangent):
    """
    Parallel transport `tangent`, a tangent vector at point `base`, along the
    geodesic in the direction `direction` (another tangent vector at point
    `base`, not necessarily unit length)
    """
    norm_direction = np.sqrt(minkowski_dot(direction, direction))
    unit_direction = direction / norm_direction
    parallel_component = minkowski_dot(tangent, unit_direction)
    unit_direction_transported = np.sinh(norm_direction) * base + np.cosh(norm_direction) * unit_direction
    return parallel_component * unit_direction_transported + tangent - parallel_component * unit_direction 
