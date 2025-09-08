import numpy as np
from utils import get_forces

point = np.dtype([
    ('x', np.float16),
    ('y', np.float16),
    ('mass', np.int32)
])

def compute_force(point1: point, point2: point) -> np.float64: # type: ignore
    f = np.float64(0.00)
    for force_func in get_forces():
        f += force_func(point1, point2)
    
    return f

