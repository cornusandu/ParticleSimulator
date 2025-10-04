#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <cstdint>


struct Point {
    double x, y;
    uint32_t mass;
    double vx, vy;
};



__device__ double2 scalar_to_2d(double scl,
    const double2 start,
    const double2 target)
{
    double2 out_vec;
    double dx = target.x - start.x;
    double dy = target.y - start.y;
    double dist = sqrt(dx * dx + dy * dy);

    if (dist < 1e-9) {
        out_vec.x = 0.0;
        out_vec.y = 0.0;
        return out_vec;
    }

    double inv_dist = 1.0 / dist;
    out_vec.x = scl * dx * inv_dist;
    out_vec.y = scl * dy * inv_dist;
    return out_vec;
}



__device__ double compute_grav_force(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double distance_squared = dx * dx + dy * dy;
    if (distance_squared == 0) {
        return 0.0;
    }
    return p1.mass * p2.mass / distance_squared;
}

__global__ void compute_new(Point *points, uint8_t iterative_steps, double dt) {
    if ((blockIdx.y * blockDim.y + threadIdx.y) >= blockIdx.x * blockDim.x + threadIdx.x) {
        return;
    }
    
    Point &p1 = points[blockIdx.x * blockDim.x + threadIdx.x];
    Point &p2 = points[blockIdx.y * blockDim.y + threadIdx.y];
    
    double force = 0.0;
    force += compute_grav_force(p1, p2);
    double2 acc1 = scalar_to_2d(force / p1.mass, make_double2(p1.x, p1.y), make_double2(p2.x, p2.y));
    double2 acc2 = scalar_to_2d(force / p2.mass, make_double2(p2.x, p2.y), make_double2(p1.x, p1.y));
    atomicAdd(&p1.vx, acc1.x * dt);
    atomicAdd(&p1.vy, acc1.y * dt);
    atomicAdd(&p2.vx, acc2.x * dt);
    atomicAdd(&p2.vy, acc2.y * dt);
}

__global__ void _compute(Point *points, uint8_t iterative_steps, double dt, int n_points) {
    const constexpr uint64_t threads = 1024;
    const uint64_t blocks = (n_points + threads - 1) / threads;

    compute_new<<<blocks, threads>>>(points, iterative_steps, dt);
}

extern "C" void compute(Point* points, int n_points, uint8_t iterative_steps, double dt) {
    _compute<<<1, 1>>>(points, iterative_steps, dt, n_points);
    cudaDeviceSynchronize();
}
