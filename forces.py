from utils import force
import numpy as np

point = np.dtype([
    ('x', np.float16),
    ('y', np.float16),
    ('mass', np.int32)
])

@force
def gravpull(point1: point, point2: point): # type: ignore
    """Gravitational pull between two points."""
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    distance_squared = dx**2 + dy**2
    if distance_squared == 0:
        return 0.0
    force = (point1['mass'] * point2['mass']) / distance_squared
    return force

@force
def closepush(point1: point, point2: point): # type: ignore
    """Repulsive force between two points when they are too close."""
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    distance_squared = dx**2 + dy**2
    distance = np.sqrt(distance_squared)
    if distance >= 5: return 0.00
    return (5 - distance) ** 1.7 * -1
