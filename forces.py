from utils import force
import numpy as np
import numba
import numba.cuda

point = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('mass', np.int32),
    ('vx', np.float64),
    ('vy', np.float64)
])

@force
@numba.njit()
def gravpull(point1: point, point2: point): # type: ignore
    """Gravitational pull between two points."""
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    distance_squared = dx**2 + dy**2
    if distance_squared == 0:
        return 0.0
    ds = np.pow(np.sqrt(distance_squared), 0.8)
    force = (point1['mass'] * point2['mass']) / (ds + 0.001)
    return force

@numba.njit()
def closepush(point1: point, point2: point): # type: ignore
    """Repulsive force between two points when they are too close."""
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    distance_squared = dx**2 + dy**2
    distance = np.sqrt(distance_squared)
    if distance >= 1: return 0.00
    return np.sqrt(1 / np.where(np.abs(distance) == 0, 0.000001, np.abs(distance))) / 40

@numba.njit()
def randomf(p1: point, p2: point):
    return np.random.uniform(-1, 1) * p1['mass'] * p2['mass']
