import random
import numpy as np
from matplotlib import pyplot as plt
import logging
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
import scipy.misc
import copy
from DAVE2pytorch import DAVE2PytorchModel
import torch
import statistics, math
import csv
from scipy.spatial.transform import Rotation as R
from ast import literal_eval
from scipy import interpolate
import PIL
from track import Track

# globals
default_color = 'green' #'Red'
default_scenario = 'jungle_rock_island'
default_spawnpoint = 'turning_radius_test'
integral = 0.0
prev_error = 0.0
throttle_setpoint = 40
steps_per_sec = 30 #100 # 36
track = None

def setup_sensors(vehicle, pos=(-0.5, 0.38, 1.3), direction=(0, 1.0, 0)):
    fov = 51 # 60 works for full lap #63 breaks on hairpin turn
    resolution = (240, 135) #(400,225) #(320, 180) #(1280,960) #(512, 512)
    front_camera = Camera(pos, direction, fov, resolution,
                          colour=True, depth=True, annotation=True)
    gforces = GForces()
    electrics = Electrics()
    damage = Damage()
    timer = Timer()

    vehicle.attach_sensor('front_cam', front_camera)
    vehicle.attach_sensor('gforces', gforces)
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('damage', damage)
    vehicle.attach_sensor('timer', timer)
    return vehicle

def ms_to_kph(wheelspeed):
    return wheelspeed * 3.6

def throttle_PID(kph, dt):
    global integral, prev_error, throttle_setpoint
    # kp = 0.001; ki = 0.00001; kd = 0.0001
    # kp = .3; ki = 0.01; kd = 0.1
    # kp = 0.15; ki = 0.0001; kd = 0.008 # worked well but only got to 39kph
    kp = 0.19; ki = 0.0001; kd = 0.008
    error = throttle_setpoint - kph
    if dt > 0:
        deriv = (error - prev_error) / dt
    else:
        deriv = 0
    integral = integral + error * dt
    w = kp * error + ki * integral + kd * deriv
    prev_error = error
    return w

# takes in 3D array of sequential [x,y]
# produces plot
def plot_deviation(trajectories, model, deflation_pattern, centerline):
    i = 0; x = []; y = []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, label="Centerline")
    for t in trajectories:
        x = []; y = []
        for point in t:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, label="Run {}".format(i))
        i += 1
    # plt.xlabel('x - axis')
    # plt.ylabel('y - axis')
    # Set a title of the current axes.
    plt.title('Trajectories with {} {}'.format(model, deflation_pattern))
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()
    plt.pause(0.1)
    return

def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment
    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892
    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)

#return distance between two 3d points
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def plot_trajectory(traj, addtl_trajs=None, title="Acceleration Trajectory", label1="car traj."):
    global track
    # plt.plot([t[0] for t in track.centerline], [t[1] for t in track.centerline], 'r-')
    # plt.plot([t[0] for t in track.roadleft], [t[1] for t in track.roadleft], 'r-')
    # plt.plot([t[0] for t in track.roadright], [t[1] for t in track.roadright], 'r-')
    x = [t[0] for t in traj]
    y = [t[1] for t in traj]
    plt.plot(x,y, 'k', label=label1)
    plt.plot([traj[0][0]], [traj[0][1]], "g^", label="spawnpoint")
    plt.plot([traj[-1][0]], [traj[-1][1]], "r8", label="endpoint")
    if addtl_trajs is not None:
        for t in addtl_trajs:
            x = [p[0] for p in t]
            y = [p[1] for p in t]
            plt.plot(x, y, 'r--', label="acceleration")
    # plt.gca().set_aspect('equal')
    # plt.axis('square')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title(title)
    plt.legend()
    plt.draw()
    plt.show()
    plt.pause(0.1)

def plot_input(timestamps, input, input_type, run_number=0):
    plt.plot(timestamps, input)
    plt.xlabel('Timestamps')
    plt.ylabel('{} input'.format(input_type))
    plt.title("{} over time".format(input_type))
    plt.savefig("Run-{}-{}.png".format(run_number, input_type))
    plt.show()
    plt.pause(0.1)

# solve for gamma where a is the corresponding vertex of gamma
def law_of_cosines(A, B, C):
    dist_AB = distance2D(A[:2], B[:2])
    dist_BC = distance2D(B[:2], C[:2])
    dist_AC = distance2D(A[:2], C[:2])
    return math.acos((math.pow(dist_AB,2)+ math.pow(dist_AC,2) -math.pow(dist_BC,2)) / (2 * dist_AB * dist_AC))

def setup_beamng(vehicle_model='etk800', model_filename="H:/GitHub/DAVE2-Keras/test-DAVE2v2-lr1e4-50epoch-batch64-lossMSE-25Ksamples-model.pt",
                 camera_pos=(-0.5, 0.38, 1.3), camera_direction=(0, 1.0, 0), pitch_euler=0.0):
    global base_filename, default_color, default_scenario,default_spawnpoint, steps_per_sec
    global integral, prev_error
    global track
    integral = 0.0
    prev_error = 0.0

    model = torch.load(model_filename, map_location=torch.device('cpu')).eval()
    # random.seed(1703)
    setup_logging()

    beamng = BeamNGpy('localhost', 64256, home='H:/BeamNG.research.v1.7.0.1clean', user='H:/BeamNG.research')

    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model,
                      licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle, pos=camera_pos, direction=camera_direction)
    spawn = track.get_spawnpoint()
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat']) #, partConfig=parts_config)
    # setup free camera
    cam_pos = (spawn['pos'][0], spawn['pos'][1], spawn['pos'][2]+70)
    eagles_eye_cam = Camera(cam_pos, (0.013892743289471, -0.015607489272952, -1.39813470840454, 0.91656774282455),
                            fov=90, resolution=(1500,1500),
                          colour=True, depth=True, annotation=True)
    scenario.add_camera(eagles_eye_cam, "eagles_eye_cam")
    # Compile the scenario and place it in BeamNG's map folder
    scenario.make(beamng)
    # Start BeamNG and enter the main loop
    bng = beamng.open(launch=True)
    bng.set_deterministic()  # Set simulator to be deterministic
    bng.set_steps_per_second(steps_per_sec)  # temporal resolution

    bng.load_scenario(scenario)
    bng.start_scenario()
    # Put simulator in pause awaiting further inputs
    bng.pause()
    assert vehicle.skt
    # bng.resume()
    return vehicle, bng, scenario, model

def quat_to_rpy(quat):
    r = R.from_quat(list(quat))
    r = r.as_euler('xyz', degrees=True)
    return tuple(r)

def vec_to_rpy(vec):
    r = R.from_rotvec(list(vec))
    r = r.as_euler('xyz', degrees=True)
    return tuple(r)

def normalize(vec):
    norm = np.linalg.norm(vec)
    normalized = np.array(vec) / norm
    return normalized

def run_test(vehicle, bng, scenario, model, vehicle_model='etk800', run_number=0,
                 device=torch.device('cuda')):
    global base_filename, default_color, default_scenario,default_spawnpoint, steps_per_sec
    global integral, prev_error, throttle_setpoint
    global track
    integral = 0.0
    prev_error = 0.0
    bng.restart_scenario()
    # collect overhead view of setup
    freecams = scenario.render_cameras()
    plt.title("freecam")
    plt.imshow(freecams['eagles_eye_cam']["colour"].convert('RGB'))
    freecams['eagles_eye_cam']["colour"].convert('RGB').save("eagles-eye-view.jpg", "JPEG")
    plt.pause(0.01)
    orig_pos = track.get_spawnpoint()['pos']
    print(vehicle.state)
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    throttle = 0.0; prev_error = throttle_setpoint; damage_prev = None; runtime = 0.0
    start_time = sensors['timer']['time']
    current_time = sensors['timer']['time']
    final_pos = None
    runtime=0; last_runtime=0
    traj = []
    while current_time <= 90:
        sensors = bng.poll_sensors(vehicle)
        current_time = sensors['timer']['time']
        runtime = sensors['timer']['time'] - start_time
        dt = runtime - last_runtime
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        # throttle = throttle_PID(kph, dt)
        vehicle.control(throttle=1.0, steering=0.0, brake=0.0)
        last_runtime = runtime
        traj.append(vehicle.state['pos'])
        print(f"{kph=}\n")
        if abs(kph - throttle_setpoint) < 1:
            print("Acceleration test completed, exiting...")
            final_pos = vehicle.state['pos']
            break
        bng.step(count=1)
    results = track.distance2D(final_pos, orig_pos)
    print(f"Acceleration distance={results}")
    print(f"Acceleration time={runtime}")
    plot_trajectory(traj)
    return results

def get_distance_traveled(traj):
    dist = 0.0
    for i in range(len(traj[:-1])):
        dist += math.sqrt(math.pow(traj[i][0] - traj[i+1][0],2) + math.pow(traj[i][1] - traj[i+1][1],2) + math.pow(traj[i][2] - traj[i+1][2],2))
    return dist

def turn_X_degrees(rot_quat, degrees=90):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + degrees
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())

def main():
    global default_scenario, integral, prev_error, track
    model_name = "model-DAVE2PytorchModel-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    model_filename = "H:/GitHub/DAVE2-Keras/{}".format(model_name)
    track = Track(scenario="jungle_rock_island", spawnpoint='turning_radius_test')
    vehicle, bng, scenario, model = setup_beamng(vehicle_model='hopper', model_filename=model_filename,
                                                 camera_pos=(-0.5, 0.38, 1.3), camera_direction=(0, 1.0, 0))

    results = run_test(vehicle, bng, scenario, model, vehicle_model='hopper', run_number=0)
    bng.close()


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    main()
