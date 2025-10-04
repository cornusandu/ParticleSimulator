# compute.pxd
cdef extern from "stdint.h":
    ctypedef unsigned int uint32_t
    ctypedef unsigned char uint8_t

cdef extern from "gpu_compute.cu":
    cdef struct Point:
        double x
        double y
        uint32_t mass
        double vx
        double vy

    void compute(Point* points, int n_points, uint8_t iterative_steps, double dt)
