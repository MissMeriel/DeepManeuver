import os.path

import cv2
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

# globals
default_color = 'green' #'Red'
default_scenario = 'industrial' #'automation_test_track'
default_spawnpoint = 'racetrackstartinggate'
integral = 0.0
prev_error = 0.0
overall_throttle_setpoint = 40
setpoint = overall_throttle_setpoint #50.0 #53.3 #https://en.wikipedia.org/wiki/Speed_limits_by_country
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []
steps_per_sec = 30 #100 # 36
training_file = 'H:/BeamNG_DeepBillboard/metas/training_runs_{}-{}1-deletelater.txt'.format(default_scenario, default_spawnpoint)

def spawn_point(scenario_locale, spawn_point ='default'):
    global lanewidth
    if scenario_locale == 'cliff':
        #return {'pos':(-124.806, 142.554, 465.489), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
        return {'pos': (-124.806, 190.554, 465.489), 'rot': None, 'rot_quat': (0, 0, 0.3826834, 0.9238795)}
    elif scenario_locale == 'west_coast_usa':
        if spawn_point == 'midhighway':
            # mid highway scenario (past shadowy parts of road)
            return {'pos': (-145.775, 211.862, 115.55), 'rot': None, 'rot_quat': (0.0032586499582976, -0.0018308814615011, 0.92652350664139, -0.37621837854385)}
        # actually past shadowy parts of road?
        #return {'pos': (95.1332, 409.858, 117.435), 'rot': None, 'rot_quat': (0.0077012465335429, 0.0036200874019414, 0.90092438459396, -0.43389266729355)}
        # surface road (crashes early af)
        elif spawn_point == 'surfaceroad1':
            return {'pos': (945.285, 886.716, 132.061), 'rot': None, 'rot_quat': (-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922)}
        # surface road 2
        elif spawn_point == 'surfaceroad2':
            return {'pos': (900.016, 959.477, 127.227), 'rot': None, 'rot_quat': (-0.046136282384396, 0.018260028213263, 0.94000166654587, 0.3375423848629)}
        # surface road 3 (start at top of hill)
        elif spawn_point == 'surfaceroad3':
            return {'pos':(873.494, 984.636, 125.398), 'rot': None, 'rot_quat':(-0.043183419853449, 2.3034785044729e-05, 0.86842048168182, 0.4939444065094)}
        # surface road 4 (right turn onto surface road) (HAS ACCOMPANYING AI DIRECTION AS ORACLE)
        elif spawn_point == 'surfaceroad4':
            return {'pos': (956.013, 838.735, 134.014), 'rot': None, 'rot_quat': (0.020984912291169, 0.037122081965208, -0.31912142038345, 0.94675397872925)}
        # surface road 5 (ramp past shady el)
        elif spawn_point == 'surfaceroad5':
            return {'pos':(166.287, 812.774, 102.328), 'rot': None, 'rot_quat':(0.0038638345431536, -0.00049926445353776, 0.60924011468887, 0.79297626018524)}
        # entry ramp going opposite way
        elif spawn_point == 'entryrampopp':
            return {'pos': (850.136, 946.166, 123.827), 'rot': None, 'rot_quat': (-0.030755277723074, 0.016458060592413, 0.37487033009529, 0.92642092704773)}
        # racetrack
        elif spawn_point == 'racetrack':
            return {'pos': (395.125, -247.713, 145.67), 'rot': None, 'rot_quat': (0, 0, 0.700608, 0.713546)}
    elif scenario_locale == 'smallgrid':
        return {'pos':(0.0, 0.0, 0.0), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
        # right after toll
        return {'pos': (-852.024, -517.391 + lanewidth, 106.620), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}
        # return {'pos':(-717.121, 101, 118.675), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
        return {'pos': (-717.121, 101, 118.675), 'rot': None, 'rot_quat': (0, 0, 0.918812, -0.394696)}
    elif scenario_locale == 'automation_test_track':
        if spawn_point == 'startingline':
            # starting line
            return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif spawn_point == 'starting line 30m down':
            # 30m down track from starting line
            return {'pos': (530.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif spawn_point == 'handlingcircuit':
            # handling circuit
            return {'pos': (-294.031, 10.4074, 118.518), 'rot': None, 'rot_quat': (0, 0, 0.708103, 0.706109)}
        elif spawn_point == 'handlingcircuit2':
            return {'pos': (-280.704, -25.4946, 118.794), 'rot': None, 'rot_quat': (-0.00862686, 0.0063203, 0.98271, 0.184842)}
        elif spawn_point == 'handlingcircuit3':
            return {'pos': (-214.929, 61.2237, 118.593), 'rot': None, 'rot_quat': (-0.00947676, -0.00484788, -0.486675, 0.873518)}
        elif spawn_point == 'handlingcircuit4':
            # return {'pos': (-180.663, 117.091, 117.654), 'rot': None, 'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
            # return {'pos': (-171.183,147.699,117.438), 'rot': None, 'rot_quat': (0.001710215350613,-0.039731655269861,0.99312973022461,-0.11005393415689)}
            return {'pos': (-173.009,137.433,116.701), 'rot': None,'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
            return {'pos': (-166.679, 146.758, 117.68), 'rot': None,'rot_quat': (0.075107827782631, -0.050610285252333, 0.99587279558182, 0.0058960365131497)}
        elif spawn_point == 'rally track':
            # rally track
            return {'pos': (-374.835, 84.8178, 115.084), 'rot': None, 'rot_quat': (0, 0, 0.718422, 0.695607)}
        elif spawn_point == 'highway':
            # highway (open, farm-like)
            return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': (0, 0, -0.704635, 0.70957)}
        elif spawn_point == 'highwayopp':
            # highway (open, farm-like)
            return {'pos': (-542.719,-251.721,117.083), 'rot': None, 'rot_quat': (0.0098941307514906,0.0096141006797552,0.72146373987198,0.69231480360031)}
        elif spawn_point == 'default':
            # default
            return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
    elif scenario_locale == 'industrial':
        if spawn_point == 'west':
            # western industrial area -- didnt work with AI Driver
            return {'pos': (237.131, -379.919, 34.5561), 'rot': None, 'rot_quat': (-0.035, -0.0181, 0.949, 0.314)}
        # open industrial area -- didnt work with AI Driver
        # drift course (dirt and paved)
        elif spawn_point == 'driftcourse':
            return {'pos': (20.572, 161.438, 44.2149), 'rot': None, 'rot_quat': (-0.003, -0.005, -0.636, 0.771)}
        # rallycross course/default
        elif spawn_point == 'rallycross':
            return {'pos': (4.85287, 160.992, 44.2151), 'rot': None, 'rot_quat': (-0.0032, 0.003, 0.763, 0.646)}
        # racetrack
        elif spawn_point == 'racetrackright':
            return {'pos': (184.983, -41.0821, 42.7761), 'rot': None, 'rot_quat': (-0.005, 0.001, 0.299, 0.954)}
        elif spawn_point == 'racetrackleft':
            return {'pos': (216.578, -28.1725, 42.7788), 'rot': None, 'rot_quat': (-0.0051, -0.003147, -0.67135, 0.74112)}
        elif spawn_point == 'racetrackstartinggate':
            return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
        elif spawn_point == "racetrackstraightaway":
            return {'pos':(262.328, -35.933, 42.5965), 'rot': None, 'rot_quat':(-0.010505940765142, 0.029969356954098, -0.44812294840813, 0.89340770244598)}
        elif spawn_point == "racetrackcurves":
            return {'pos':(215.912,-243.067,45.8604), 'rot': None, 'rot_quat':(0.029027424752712,0.022241719067097,0.98601061105728,0.16262225806713)}

def setup_sensors(vehicle, pos=(-0.5, 0.38, 1.3), direction=(0, 1.0, 0)):
    # Set up sensors
    # pos = (-0.3, 1, 1.0) # default
    # pos = (-0.5, 2, 1.0) #center edge of hood
    # pos = (-0.5, 1, 1.0)  # center middle of hood
    # pos = (-0.5, 0.4, 1.0)  # dashboard
    # pos = (-0.5, 0.38, 1.5) # roof
    # pos = (-0.5, 0.38, 1.3) # windshield
    # pos = (-0.5, 0.38, 1.7) # above windshield
    # direction = (0, 1.0, -0.275)
    fov = 51 # 60 works for full lap #63 breaks on hairpin turn
    resolution = (240, 135) #(400,225) #(320, 180) #(1280,960) #(512, 512)
    front_camera = Camera(pos, direction, fov, resolution,
                          colour=True, depth=True, annotation=True)

    gforces = GForces()
    electrics = Electrics()
    damage = Damage()
    timer = Timer()

    # Attach them
    vehicle.attach_sensor('front_cam', front_camera)
    vehicle.attach_sensor('gforces', gforces)
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('damage', damage)
    vehicle.attach_sensor('timer', timer)
    return vehicle

def ms_to_kph(wheelspeed):
    return wheelspeed * 3.6

def throttle_PID(kph, dt):
    global integral, prev_error, setpoint
    # kp = 0.001; ki = 0.00001; kd = 0.0001
    # kp = .3; ki = 0.01; kd = 0.1
    # kp = 0.15; ki = 0.0001; kd = 0.008 # worked well but only got to 39kph
    kp = 0.19; ki = 0.0001; kd = 0.008
    error = setpoint - kph
    if dt > 0:
        deriv = (error - prev_error) / dt
    else:
        deriv = 0
    integral = integral + error * dt
    w = kp * error + ki * integral + kd * deriv
    prev_error = error
    return w

def diff_damage(damage, damage_prev):
    new_damage = 0
    if damage is None or damage_prev is None:
        return 0
    new_damage = damage['damage'] - damage_prev['damage']
    return new_damage

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

def dist_from_line(centerline, point):
    a = [[x[0],x[1]] for x in centerline[:-1]]
    b = [[x[0],x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist

def calc_deviation_from_center(centerline, traj):
    dists = []
    for point in traj:
        dist = dist_from_line(centerline, point)
        dists.append(min(dist))
    stddev = statistics.stdev(dists)
    return stddev

def plot_racetrack_roads(roads, bng):
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp = []
        y_temp = []
        dont_add = False
        for edge in road_edges:
            if edge['middle'][0] < 100:
                dont_add = True
                break
            if edge['middle'][1] < -300 or edge['middle'][1] > 0:
                dont_add = True
                break
            if not dont_add:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
        if not dont_add:
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
    plt.legend()
    plt.show()
    plt.pause(0.001)

def road_analysis(bng):
    global centerline, roadleft, roadright
    plot_racetrack_roads(bng.get_roads(), bng)
    # get relevant road
    edges = bng.get_road_edges('7983')
    actual_middle = [edge['middle'] for edge in edges]
    roadleft = [edge['left'] for edge in edges]
    roadright = [edge['right'] for edge in edges]
    adjusted_middle = [np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle']))/4.0 for edge in edges]
    centerline = actual_middle
    return actual_middle, adjusted_middle

def plot_trajectory(traj, title="Trajectory", label1="car traj."):
    global centerline, roadleft, roadright
    plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'r-')
    plt.plot([t[0] for t in roadleft], [t[1] for t in roadleft], 'r-')
    plt.plot([t[0] for t in roadright], [t[1] for t in roadright], 'r-')
    x = [t[0] for t in traj]
    y = [t[1] for t in traj]
    plt.plot(x,y, 'b--', label=label1)
    # plt.gca().set_aspect('equal')
    # plt.axis('square')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title(title)
    plt.legend()
    plt.draw()
    plt.savefig(f"{title}.jpg")
    plt.show()
    plt.pause(0.1)

def create_ai_line_from_road(spawn, bng):
    line = []; points = []; point_colors = []; spheres = []; sphere_colors = []
    middle = road_analysis(bng)
    middle_end = middle[:3]
    middle = middle[3:]
    middle.extend(middle_end)
    traj = []
    with open("centerline_lap_data.txt", 'w') as f:
        for i,p in enumerate(middle[:-1]):
            f.write("{}\n".format(p))
            # interpolate at 1m distance
            if distance(p, middle[i+1]) > 1:
                y_interp = scipy.interpolate.interp1d([p[0], middle[i+1][0]], [p[1], middle[i+1][1]])
                num = abs(int(middle[i+1][0] - p[0]))
                xs = np.linspace(p[0], middle[i+1][0], num=num, endpoint=True)
                ys = y_interp(xs)
                for x,y in zip(xs,ys):
                    traj.append([x,y])
                    line.append({"x":x, "y":y, "z":p[2], "t":i * 10})
                    points.append([x, y, p[2]])
                    point_colors.append([0, 1, 0, 0.1])
                    spheres.append([x, y, p[2], 0.25])
                    sphere_colors.append([1, 0, 0, 0.8])
            else:
                traj.append([p[0],p[1]])
                line.append({"x": p[0], "y": p[1], "z": p[2], "t": i * 10})
                points.append([p[0], p[1], p[2]])
                point_colors.append([0, 1, 0, 0.1])
                spheres.append([p[0], p[1], p[2], 0.25])
                sphere_colors.append([1, 0, 0, 0.8])
            plot_trajectory(traj, "Points on Script So Far")
    plot_trajectory(traj, "Planned traj.")
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng

def plot_input(timestamps, input, input_type, run_number=0):
    plt.plot(timestamps, input)
    plt.xlabel('Timestamps')
    plt.ylabel('{} input'.format(input_type))
    plt.title("{} over time".format(input_type))
    plt.savefig("Run-{}-{}.png".format(run_number, input_type))
    plt.show()
    plt.pause(0.1)

def create_ai_line_from_road_with_interpolation(spawn, bng):
    global centerline, remaining_centerline, centerline_interpolated
    line = []; points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
    actual_middle, adjusted_middle = road_analysis(bng)
    middle_end = adjusted_middle[:3]
    middle = adjusted_middle[3:]
    temp = [list(spawn['pos'])]; temp.extend(middle); middle = temp
    middle.extend(middle_end)
    remaining_centerline = copy.deepcopy(middle)
    timestep = 0.1; elapsed_time = 0; count = 0
    # set up adjusted centerline
    for i,p in enumerate(middle[:-1]):
        # interpolate at 1m distance
        if distance(p, middle[i+1]) > 1:
            y_interp = interpolate.interp1d([p[0], middle[i+1][0]], [p[1], middle[i+1][1]])
            num = int(distance(p, middle[i+1]))
            xs = np.linspace(p[0], middle[i+1][0], num=num, endpoint=True)
            ys = y_interp(xs)
            for x,y in zip(xs,ys):
                traj.append([x,y])
        else:
            elapsed_time += distance(p, middle[i+1]) / 12
            traj.append([p[0],p[1]])
            linedict = {"x": p[0], "y": p[1], "z": p[2], "t": elapsed_time}
            line.append(linedict)
            count += 1
    # set up debug line
    for i,p in enumerate(actual_middle[:-1]):
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.25])
        sphere_colors.append([1, 0, 0, 0.8])
        count += 1
    print("spawn point:{}".format(spawn))
    print("beginning of script:{}".format(middle[0]))
    plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
    # centerline = copy.deepcopy(traj)
    remaining_centerline = copy.deepcopy(traj)
    centerline_interpolated = copy.deepcopy(traj)
    for i in range(4):
        centerline.extend(copy.deepcopy(centerline))
        remaining_centerline.extend(copy.deepcopy(remaining_centerline))
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng

# track is approximately 12.50m wide
# car is approximately 1.85m wide
def has_car_left_track(vehicle_pos, vehicle_bbox, bng):
    global centerline_interpolated
    # get nearest road point
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    # print("Distance from center of road:", min(distance_from_centerline))
    return min(distance_from_centerline) > 9.0

def find_width_of_road(bng):
    edges = bng.get_road_edges('7983')
    left_edge = [edge['left'] for edge in edges]
    right_edge = [edge['right'] for edge in edges]
    middle = [edge['middle'] for edge in edges]
    width = distance(left_edge[0], middle[0]) + distance(right_edge[0], middle[0])
    print("width of road:", (width))
    return width

def add_qr_cubes(scenario):
    global qr_positions
    qr_positions = []
    with open('qr_box_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
            box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
                                  rot_quat=rot_quat, scale=(1,1,1), JBeam = 'qrbox2', datablock="default_vehicle")
            scenario.add_object(box)

def setup_beamng(vehicle_model='etk800', model_filename="H:/GitHub/DAVE2-Keras/test-DAVE2v2-lr1e4-50epoch-batch64-lossMSE-25Ksamples-model.pt", camera_pos=(-0.5, 0.38, 1.3), camera_direction=(0, 1.0, 0), pitch_euler=0.0):
    global base_filename, default_color, default_scenario,default_spawnpoint, steps_per_sec
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0

    model = torch.load(model_filename, map_location=torch.device('cpu')).eval()
    print(model)
    print(model.input_shape)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    random.seed(1703)
    setup_logging()

    beamng = BeamNGpy('localhost', 64256, home='H:/BeamNG.research.v1.7.0.1clean', user='H:/BeamNG.research')

    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model,
                      licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle, pos=camera_pos, direction=camera_direction)
    spawn = spawn_point(default_scenario, default_spawnpoint)
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat']) #, partConfig=parts_config)
    add_barriers(scenario)
    add_qr_cubes(scenario)
    # setup free camera
    eagles_eye_cam = Camera((221.854, -128.443, 165.5),
                            (0.013892743289471, -0.015607489272952, -1.39813470840454, 0.91656774282455),
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
    ai_line, bng = create_ai_line_from_road_with_interpolation(spawn, bng)
    # Put simulator in pause awaiting further inputs
    bng.pause()
    assert vehicle.skt
    # bng.resume()
    # find_width_of_road(bng)
    return vehicle, bng, scenario, model

def run_scenario(vehicle, bng, scenario, model, vehicle_model='etk800',
                 camera_pos=(-0.5, 0.38, 1.3), camera_direction=(0, 1.0, 0), pitch_euler=0.0, run_number=0,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    global base_filename, default_color, default_scenario,default_spawnpoint, steps_per_sec
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0
    bng.restart_scenario()
    # collect overhead view of setup
    freecams = scenario.render_cameras()
    plt.title("freecam")
    plt.imshow(freecams['eagles_eye_cam']["colour"].convert('RGB'))
    freecams['eagles_eye_cam']["colour"].convert('RGB').save("eagles-eye-view.jpg", "JPEG")
    plt.pause(0.01)

    # perturb vehicle
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    image = sensors['front_cam']['colour'].convert('RGB')
    pitch = vehicle.state['pitch'][0]
    roll = vehicle.state['roll'][0]
    z = vehicle.state['pos'][2]
    spawn = spawn_point(default_scenario, default_spawnpoint)

    # print("VEHICLE BOUNDING BOX:{}".format(vehicle.get_bbox()))
    wheelspeed = 0.0; throttle = 0.0; prev_error = setpoint; damage_prev = None; runtime = 0.0
    kphs = []; traj = []; steering_inputs = []; throttle_inputs = []; timestamps = []
    damage = None; overall_damage = 0.0
    final_img = None
    total_loops = 0; total_imgs = 0; total_predictions = 0
    start_time = sensors['timer']['time']

    writedir = "H:/GitHub/Defects4DL/industrial-track-lap"
    if not os.path.isdir(writedir):
        os.mkdir(writedir)
    with open(f"{writedir}/data.txt", "w") as f:
        f.write(f"IMG,PREDICTION,POSITION,ORIENTATION,KPH,STEERING_ANGLE_CURRENT\n")

        while overall_damage <= 0:
            # collect images
            vehicle.update_vehicle()
            sensors = bng.poll_sensors(vehicle)
            image = sensors['front_cam']['colour'].convert('RGB')
            print(f"{type(image)=}")
            cv2.imshow('car view', np.array(image)[:, :, ::-1])
            cv2.waitKey(1)
            total_imgs += 1
            kph = ms_to_kph(sensors['electrics']['wheelspeed'])
            dt = (sensors['timer']['time'] - start_time) - runtime
            prediction = model(model.process_image(image).to(device))
            print(f"{prediction=}")
            runtime = sensors['timer']['time'] - start_time

            # control params
            steering = float(prediction[0][0])
            total_predictions += 1
            position = str(vehicle.state['pos']).replace(",", " ")
            orientation = str(vehicle.state['dir']).replace(",", " ")

            image.save(f"{writedir}/sample-{total_imgs:05d}.jpg", "JPEG")
            f.write(f"sample-{total_imgs:05d}.jpg,{prediction.item()},{position},{orientation},{kph},{sensors['electrics']['steering']}\n")
            if abs(steering) > 0.2:
                setpoint = 30
            else:
                setpoint = 40
            throttle = throttle_PID(kph, dt)
            vehicle.control(throttle=throttle, steering=steering, brake=0.0)

            steering_inputs.append(steering)
            throttle_inputs.append(throttle)
            timestamps.append(runtime)

            damage = sensors['damage']
            overall_damage = damage["damage"]
            new_damage = diff_damage(damage, damage_prev)
            damage_prev = damage
            vehicle.update_vehicle()
            traj.append(vehicle.state['pos'])

            kphs.append(ms_to_kph(wheelspeed))
            total_loops += 1
            final_img = image

            if new_damage > 0.0:
                print("New damage={}, exiting...".format(new_damage))
                break
            bng.step(1, wait=True)

            # if runtime > 300:
            #     print("Exited after 5 minutes successful runtime")
            #     break

            if distance(spawn['pos'], vehicle.state['pos']) < 5 and runtime > 10:
                print("Completed one lap, exiting...")
                reached_start = True
                break

            outside_track = has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng)
            if outside_track:
                print("Left track, exiting...")
                break

    cv2.destroyAllWindows()

    print("Total predictions: {} \nexpected predictions ={}*{}={}".format(total_predictions, round(runtime,3), steps_per_sec, round(runtime*steps_per_sec,3)))
    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj, 'pitch': round(pitch,3),
               'roll':round(roll,3), "z":round(z,3), 'final_img':final_img,
               'camera_direction':camera_direction, 'camera_position':camera_pos,
               'deviation':deviation
               }
    return results

def get_distance_traveled(traj):
    dist = 0.0
    for i in range(len(traj[:-1])):
        dist += math.sqrt(math.pow(traj[i][0] - traj[i+1][0],2) + math.pow(traj[i][1] - traj[i+1][1],2) + math.pow(traj[i][2] - traj[i+1][2],2))
    return dist

def turn_90(rot_quat):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + 90
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())

def add_barriers(scenario):
    barrier_locations = []
    with open('industrial_racetrack_barrier_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            rot_quat = turn_90(rot_quat)
            # barrier_locations.append({'pos':pos, 'rot_quat':rot_quat})
            # add barrier to scenario
            ramp = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
            scenario.add_object(ramp)

def main():
    global base_filename, default_color, default_scenario, setpoint, integral
    global prev_error, centerline
    # camera_zs = [1.3] #[0.87, 1.07, 1.3, 1.51, 1.73]
    # with open(training_file, 'w') as f:
    #     f.write("CAMERA_DIR,CAMERA_PITCH_EULER,CAMERA_POS,RUNTIME,DISTANCE,AVG_STDEV,RUN_SCORE\n")
    #     for z in camera_zs:
    camera_pos = (-0.5, 0.38, 1.3)
    camera_dir = (0, 1.0, 0)
    model_name = "model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    model_name = "model-DAVE2PytorchModel-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    #"model-DAVE2PytorchModel-lr1e4-100epoch-batch64-lossMSE-76Ksamples-losslimit1e-3-INDUSTRIALandHIROCHIonly-135x240-noiseflip3.pt"
    model_name = "model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-59Ksamples-135x240-NOROBUSTIFICATION.pt" #"model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-59Ksamples-135x240-noiseflipblur.pt"
    model_filename = "H:/GitHub/DAVE2-Keras/{}".format(model_name)
    #model-DAVE2PytorchModel-lr1e4-50epoch-batch64-lossMSE-25Ksamples-losslimit1e-3-racetrackanddriftcourseandHIROCHI3.pt
    # "H:/GitHub/DAVE2-Keras/model-DAVE2PytorchModel-lr1e4-50epoch-batch64-lossMSE-25Ksamples-losslimit1e-3-rerunofnew25K.pt"
    # "H:/GitHub/DAVE2-Keras/test-7-trad-50epochs-64batch-1e4lr-ORIGDATASET-singleoutput-model-epoch-43.pt",
    # "H:/GitHub/DAVE2-Keras/test-dave2v1-lr1e4-50epoch-batch64-lossMSE-25Ksamples-model-epoch-49.pt",
    # "H:/GitHub/DAVE2-Keras/model-DAVE2PytorchModel-lr1e4-50epoch-batch64-lossMSE-25Ksamples-losslimit1e-3.pt",
    # "H:/GitHub/DAVE2-Keras/test-DAVE2v2-lr1e4-50epoch-batch64-lossMSE-25Ksamples-model.pt",
    model_name = "H:/GitHub/DAVE2-Keras/model-DAVE2PytorchModel-lr1e6-50epoch-batch64-lossMSE-25Ksamples-losslimit1e-3-withswerving5.pt",
    #"model-DAVE2PytorchModel-lr1e4-50epoch-batch64-lossMSE-25Ksamples-losslimit1e-3-racetrackanddriftcourseandHIROCHI2.pt"
    vehicle, bng, scenario, model = setup_beamng(vehicle_model='hopper', model_filename=model_filename, camera_pos=camera_pos,
                                                camera_direction=camera_dir)

    for i in range(1):
        # print("\nRun {} for camera z={}".format(i, z))
        results = run_scenario(vehicle, bng, scenario, model, vehicle_model='hopper', camera_pos=camera_pos, camera_direction=camera_dir, run_number=i)
        results['distance'] = get_distance_traveled(results['traj'])
        # print("Writing trajectory to file...")
        # with open(f"{model._get_name()}-lap-trajectory.txt", "w") as f:
        #     for p in results['traj']:
        #         f.write(f"{p}\n")
        # print(f"Trajectory written to file {model._get_name()}-lap-trajectory.txt")
        # sample_entry = f"runtime={results['runtime']:.2f}-dist_travelled={results['distance']:.2f}-stdev={results['deviation"]:.2f}")
        plot_trajectory(results['traj'], f"{model._get_name()}-40kph-runtime{results['runtime']:.2f}-distance{results['distance']:.2f}")
        # f.write("{}\n".format(sample_entry))
    bng.close()


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    main()
