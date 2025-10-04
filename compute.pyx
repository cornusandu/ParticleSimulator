# compute.pyx
# distutils: language = c++
# cython: language_level=3

from compute cimport Point, compute, uint8_t
import numpy as np
cimport numpy as np


cdef class PyPoint:
    cdef Point p
    def __init__(self, double x=0, double y=0, unsigned int mass=1, double vx=0, double vy=0):
        self.p.x = x
        self.p.y = y
        self.p.mass = mass
        self.p.vx = vx
        self.p.vy = vy
    def __repr__(self):
        return f"Point(x={self.p.x}, y={self.p.y}, vx={self.p.vx}, vy={self.p.vy})"


def gpu_compute(list py_points, double dt, int steps=1):
    """Compute gravitational interactions using CUDA."""
    cdef int n = len(py_points)
    cdef Point[::1] arr = np.empty(n, dtype=np.dtype([
        ('x', np.float64), ('y', np.float64),
        ('mass', np.uint32), ('vx', np.float64), ('vy', np.float64)
    ])).view(Point)

    for i in range(n):
        arr[i] = py_points[i].p

    compute(&arr[0], n, <uint8_t>steps, dt)

    for i in range(n):
        py_points[i].p = arr[i]

    return py_points
