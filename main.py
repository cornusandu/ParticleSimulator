import numpy as np
from utils import get_forces, scalar_to_2d
from vispy import app, scene
from vispy.scene import visuals
import time
import os
import forces
import numba
import numba.cuda

import vispy.app
vispy.app.use_app('glfw')  # or 'pyqt6', 'pyside6'


point = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('mass', np.int32),
    ('vx', np.float64),
    ('vy', np.float64)
])

fast_forward = 1

@numba.jit(nopython=False, parallel=True)
def compute_force(point1: point, point2: point, force_funcs) -> np.ndarray: # type: ignore
    f = np.zeros(2, dtype=np.float64)
    for force_func in force_funcs:
        f += scalar_to_2d(force_func(point1, point2), np.array([point1['x'], point1['y']]), np.array([point2['x'], point2['y']]))
    
    return f

@numba.njit()
def compute_acc(p: point, force: np.ndarray) -> np.ndarray:
    return force / p['mass']

@numba.njit()
def compute_speed(vel: np.ndarray, acc: np.ndarray) -> np.ndarray:  # ([vx, vy], acc, dt) -> [vx, vy]
    return vel + acc

points = np.zeros(10, dtype=point)   # start with capacity for 10
n_points = 0                         # how many are actually used
scatter = None

def add_point(p: point):
    global points, n_points
    if n_points >= len(points):
        # grow capacity
        points = np.resize(points, len(points) + 10)
    points[n_points] = p
    n_points += 1

def setup():
    global points, point
    for i in range(32):
        p = np.zeros(1, dtype=point)[0]   # create one structured record
        p['x'] = np.random.uniform(-20, 20)
        p['y'] = np.random.uniform(-20, 20)
        p['mass'] = np.random.randint(1, 30)
        p['vx'] = np.random.uniform(-1, 1)
        p['vy'] = np.random.uniform(-1, 1)
        add_point(p)

@numba.njit
def get_radius(p: point):
    return np.pow(p['mass'], .9801)

#@numba.jit(nopython=False, parallel=True)
def update(dt, do_forces=True, fast_forward=False, forces=[]):
    global n_points, points

    if fast_forward:
        # run forces once with dt*2
        if do_forces:
            for i in range(n_points):
                for j in range(i):
                    force = compute_force(points[i], points[j], forces)
                    acc1 = force / points[i]['mass']
                    points[i]['vx'] += acc1[0] * dt*2
                    points[i]['vy'] += acc1[1] * dt*2

                    acc2 = -force / points[j]['mass']
                    points[j]['vx'] += acc2[0] * dt*2
                    points[j]['vy'] += acc2[1] * dt*2

        # integrate positions 4 times (skip extra force calcs)
        for _ in range(2):
            for i in range(n_points):
                points[i]['x'] += points[i]['vx'] * dt*2
                points[i]['y'] += points[i]['vy'] * dt*2
    else:
        if do_forces:
            for i in range(n_points):
                for j in range(i):
                    force = compute_force(points[i], points[j], forces)
                    acc1 = force / points[i]['mass']
                    points[i]['vx'] += acc1[0] * dt
                    points[i]['vy'] += acc1[1] * dt

                    acc2 = -force / points[j]['mass']
                    points[j]['vx'] += acc2[0] * dt
                    points[j]['vy'] += acc2[1] * dt

        for i in range(n_points):
            points[i]['x'] += points[i]['vx'] * dt
            points[i]['y'] += points[i]['vy'] * dt

def render(canvas):
    global scatter, points, n_points
    coords = np.array([[points[i]['x'], points[i]['y']] for i in range(n_points)])
    sizes = np.pow(np.array([points[i]['mass'] for i in range(n_points)], dtype=float), .98)
    scatter.set_data(coords, face_color='red', size=sizes, symbol='o')
    canvas.events.draw()

def main():
    global scatter, fast_forward, points

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    view.add(scatter)

    setup()
    render(canvas)

    view.camera = scene.cameras.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(-10, 10), y=(-10, 10))

    last_time = time.time()

    @canvas.events.key_press.connect
    def on_key(event):
        global fast_forward
        if event.key == 'Right':
            fast_forward = 4
        elif event.key == 'Up':
            fast_forward = 64
        elif event.key == 'Down':
            fast_forward = 4096 * 4

    @canvas.events.key_release.connect
    def on_key_release(event):
        global fast_forward
        if event.key == 'Right':
            fast_forward = 1
        elif event.key == 'Up':
            fast_forward = 1
        elif event.key == 'Down':
            fast_forward = 1

    f = get_forces()

    def con(*_, **__):
        global fast_forward
        nonlocal last_time
        nonlocal f
        delta = time.time() - last_time
        if fast_forward > 1:
            # simulate 8x faster:
            # 1) do force update with dt*2
            update((time.time()-last_time)*fast_forward, do_forces=True, forces=f)
            # 2) skip forces, integrate positions only (3 more times)
            for _ in range(3):
                update((time.time()-last_time)*fast_forward, do_forces=False, forces=f)
        else:
            update(time.time()-last_time, do_forces=True, forces=f)

        render(canvas)

        canvas.title = f"Particle Simulator - FPS: {canvas.fps:.9f}"

        last_time = time.time()

    timer = app.Timer(connect=con, start=True)
    app.run()
        
main()
