import numpy as np
import numba
import numba.cuda

__forces = []

def force(func):
    __forces.append(func)
    return func

def get_forces():
    return __forces.copy()

@numba.njit()
def scalar_to_2d(scl: np.float64, start: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Converts a scalar magnitude into a 2D vector pointing from `start` toward `target`.
    
    Parameters
    ----------
    scl : np.float64
        The scalar magnitude (e.g., force, speed).
    start : np.ndarray
        The starting point (x, y).
    target : np.ndarray
        The target point (x, y).
    
    Returns
    -------
    np.ndarray
        A 2D vector (x, y) with magnitude = scl pointing toward target.
    """
    
    # Direction vector
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    
    # Length of direction vector
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 1e-9:
        return np.zeros(2, dtype=np.float64)   # no direction if points overlap
    
    # Normalize direction vector and scale by `scl`
    unit_vec = np.array([dx / dist, dy / dist])
    return scl * unit_vec
