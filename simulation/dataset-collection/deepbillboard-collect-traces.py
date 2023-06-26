import random
import numpy as np
from matplotlib import pyplot as plt

from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject, ProceduralMesh, ProceduralCube
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

import scipy.misc
import copy
from DAVE2 import DAVE2Model
import statistics, math
import json, csv
from scipy.spatial.transform import Rotation as R
from ast import literal_eval
from scipy import interpolate
import cv2
from cv2 import QRCodeDetector
from PIL import Image, ImageOps
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from kraken import binarization
import logging
from matplotlib import cm
from skimage import color
from skimage import io
from skimage import util


# globals
default_color = 'White' #'Red'
default_scenario = 'industrial' #'automation_test_track'
default_spawnpoint = 'racetrackstartinggate'
integral = 0.0
prev_error = 0.0
setpoint = 40 #50.0 #53.3 #https://en.wikipedia.org/wiki/Speed_limits_by_country
lanewidth = 3.75 #2.25
centerline = []
centerline_interpolated = []
steps_per_sec = 30 #60 #100 # 36
base_filename = 'H:/BeamNG_DeepBillboard2/traces/' #'H:/BeamNG_DeepBillboard2/traces/training_runs_{}-{}'.format(default_scenario, default_spawnpoint)
# training_file = 'H:/BeamNG_DeepBillboard2/traces/coordinates.txt' #.format(default_scenario, default_spawnpoint)

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
            return {'pos': (216.627, -38.8136, 42.7541), 'rot': None,
                    'rot_quat': (0.0053605171851814, 0.0076489462517202, 0.86981534957886, -0.49328899383545)}
            # return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
        elif spawn_point == "racetrackstraightaway":
            return {'pos':(262.328, -35.933, 42.5965), 'rot': None, 'rot_quat':(-0.010505940765142, 0.029969356954098, -0.44812294840813, 0.89340770244598)}
        elif spawn_point == "racetrackcurves":
            return {'pos':(215.912,-243.067,45.8604), 'rot': None, 'rot_quat':(0.029027424752712,0.022241719067097,0.98601061105728,0.16262225806713)}

def setup_sensors(vehicle, pos=(-0.5, 0.38, 1.3), direction=(0, 1.0, 0)):
    # Set up sensors
    # pos = (-0.3, 1, 1.0) # default
    #pos = (-0.5, 2, 1.0) #center edge of hood
    # pos = (-0.5, 1, 1.0)  # center middle of hood
    # pos = (-0.5, 0.4, 1.0)  # dashboard
    # pos = (-0.5, 0.38, 1.5) # roof
    # pos = (-0.5, 0.38, 1.3) # windshield
    # direction = (0, 1.0, 0)
    # fov = 50
    # resolution = (200, 150) #(1280,960) #(512, 512)
    fov = 51 # 60 works for full lap #63 breaks on hairpin turn
    resolution = (16*15, 9*15) #(320, 180) #(1280,960) #(512, 512)
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

def threeD_to_twoD(arr):
    return [[x[0],x[1]] for x in arr]

#return distance between two 3d points
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def dist_from_line(centerline, point):
    a = threeD_to_twoD(centerline[:-1])
    b = threeD_to_twoD(centerline[1:])
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

def road_analysis(bng):
    roads = bng.get_roads()
    # get relevant road
    edges = bng.get_road_edges('7983')
    middle = [edge['middle'] for edge in edges]
    return middle

def intake_ai_lap_poses(filename="ai_lap_data.txt"):
    global centerline
    lap_traj = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            # print(line)
            line = literal_eval(line)
            lap_traj.append(line)
    centerline = lap_traj
    plot_trajectory(lap_traj)
    return lap_traj

def create_ai_line(bng, filename="ai_lap_data.txt"):
    line = []; points = []; point_colors = []; spheres = []; sphere_colors = []
    poses = intake_ai_lap_poses(filename)
    for i,p in enumerate(poses):
        if p[1] > -50:
            p[1] += 3
        if p[0] > 300:
            p[0] += 3
        if p[1] < -225:
            p[0] -= 3
        if i % 5 == 0:
            line.append({"x":p[0], "y":p[1], "z":p[2], "t":0.5 * i})
            points.append(p)
            point_colors.append([0, 1, 0, 0.1])
            spheres.append([p[0], p[1], p[2], 0.25])
            sphere_colors.append([np.sin(np.radians(10)), 0, 0, 0.8])
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=1)
    return line, bng

def create_ai_line_from_centerline(bng):
    line = []; points = []; point_colors = []; spheres = []; sphere_colors = []
    poses = intake_ai_lap_poses("frankenstein_lap_data.txt")
    # s = interpolate.InterpolatedUnivariateSpline([p[0] for p in poses], [p[1] for p in poses])
    count = 1
    for i,p in enumerate(poses):
        # interpolate
        # y_interp = scipy.interpolate.interp1d([p[0], poses[i+1][0]], [p[1], poses[i+1][1]])
        # num = abs(int(poses[i+1][0] - p[0]))
        # xs = np.linspace(p[0], poses[i+1][0], num=num, endpoint=True)
        # ys = y_interp(xs)
        # for x,y in zip(xs,ys):
        line.append({"x":p[0], "y":p[1], "z":p[2], "t":.1 * count})
        count += 1
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.25])
        sphere_colors.append([1, 0, 0, 0.8])
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return line, bng

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
            # plot_trajectory(traj, "Points on Script So Far")
    # print("points in centerline:{}".format(len(middle)))
    # ai_line = create_ai_line(bng)
    # print("points in ai line:{}".format(len(ai_line)))
    # print("spawn point:{}".format(spawn))
    # print("beginning of script:{}".format(middle[0]))
    plot_trajectory(traj, "Points on Script (Final)")
    # bng.add_debug_line(points, point_colors,
    #                    spheres=spheres, sphere_colors=sphere_colors,
    #                    cling=True, offset=0.1)
    return line, bng

def plot_input(timestamps, input, input_type, run_number=0):
    plt.plot(timestamps, input)
    plt.xlabel('Timestamps')
    plt.ylabel('{} input'.format(input_type))
    # Set a title of the current axes.
    plt.title("{} over time".format(input_type))
    plt.savefig("Run-{}-{}.png".format(run_number, input_type))
    plt.show()
    plt.pause(0.1)

def process_csv_for_lap_data(filename):
    global path_to_trainingdir
    hashmap = []
    timestamps = []; steerings = []; throttles = []
    with open(filename) as csvfile:
        metadata = csv.reader(csvfile, delimiter=',')
        next(metadata)
        for row in metadata:
            steerings.append(float(row[2]))
            throttles.append(float(row[3]))
            timestamps.append(float(row[1]))
            # imgfile = row[0].replace("\\", "/")
            # hashmap[i] = row[1:]
    return timestamps, steerings, throttles

def plot_one_lap_of_steering():
    filename = 'H:/BeamNG_DAVE2_racetracks_all/training_images_industrial-racetrackstartinggate0/data.csv'
    x,y_steer, y_throttle = process_csv_for_lap_data(filename)
    plt.plot(x[:1492], y_steer[:1492])
    plt.title("Steering over one lap")
    plt.show()
    plt.pause(0.01)
    print(len([i for i in y_steer if i > 0.1]))

def plot_trajectory(traj, title="Trajectory", run_number=0):
    global centerline
    x = [t[0] for t in traj]
    y = [t[1] for t in traj]
    plt.plot(x,y, 'bo', label="AI behavior")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'r+', label="AI line script")
    # Set a title of the current axes.
    plt.title(title)
    plt.legend()
    plt.savefig("Run-{}-traj.png".format(run_number))
    # Display a figure.
    plt.show()
    plt.pause(0.1)

def create_ai_line_from_road_with_interpolation():
    global centerline, centerline_interpolated
    line = []; points = []; traj = []
    middle = copy.deepcopy(centerline)
    middle_end = middle[:3]
    middle = middle[3:]
    middle.extend(middle_end)
    timestep = 0.1; elapsed_time = 0; count = 0
    for i,p in enumerate(middle[:-1]):
        # interpolate at 1m distance
        if distance(p, middle[i+1]) > 1:
            y_interp = interpolate.interp1d([p[0], middle[i+1][0]], [p[1], middle[i+1][1]])
            num = int(distance(p, middle[i+1]))
            xs = np.linspace(p[0], middle[i+1][0], num=num, endpoint=True)
            ys = y_interp(xs)
            for x,y in zip(xs,ys):
                traj.append([x,y])
                line.append({"x":x, "y":y, "z":p[2], "t":i * timestep})
                points.append([x, y, p[2]])
        else:
            elapsed_time += distance(p, middle[i+1]) / 12
            traj.append([p[0],p[1]])
            linedict = {"x": p[0], "y": p[1], "z": p[2], "t": elapsed_time}
            print(linedict)
            line.append(linedict)
            points.append([p[0], p[1], p[2]])
            count += 1
    centerline_interpolated = points

# track is approximately 12.50m wide
# car is approximately 1.85m wide
def has_car_left_track(vehicle_pos, vehicle_bbox, bng):
    global centerline_interpolated
    # get relevant road
    # edges = bng.get_road_edges('7983')
    # left_edge = [edge['left'] for edge in edges]
    # right_edge = [edge['right'] for edge in edges]
    # middle = [edge['middle'] for edge in edges]
    # get nearest road point
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    return min(distance_from_centerline) > 8

def find_width_of_road(bng):
    edges = bng.get_road_edges('7983')
    left_edge = [edge['left'] for edge in edges]
    right_edge = [edge['right'] for edge in edges]
    middle = [edge['middle'] for edge in edges]
    dist1 = distance(left_edge[0], middle[0])
    dist2 = distance(right_edge[0], middle[0])
    print("width of road:", (dist1+dist2))
    return dist1+dist2

def get_qr_corners_from_image(image):
    # Convert RGB to BGR
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # scale = 3.0
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # image = cv2.resize(image, (width, height))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    detector = QRCodeDetector()
    corners = detector.detect(gray)
    print("corners using cv2:", corners)
    if corners[1] is not None:
        upper_left = corners[1][0][3]
        lower_right = corners[1][0][1]
        cv2.rectangle(gray, (upper_left[0], upper_left[1]), (lower_right[0], lower_right[1]), (0, 255, 0), 2)
    # cv2.imshow('image', gray)
    # cv2.waitKey(0)

    # find the barcodes in the image and decode each of the barcodes
    barcodes = pyzbar.decode(image)
    print("barcodes using pyzbar:",[code.rect for code in barcodes])


    # # ret, bw_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # ret, bw_im = cv2.threshold(blur , 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #
    # # zbar
    # # bw_im = binarization.nlbin(image)
    # # bw_im.show()
    # # cv2.imshow('image',bw_im)
    # # zbar
    # corners = decode(image, symbols=None) #[ZBarSymbol.QRCODE])
    # #
    # print("corners:", corners)
    # scale = 3.0
    # width = int(image.shape[1] * scale)
    # height = int(image.shape[0] * scale)
    # image = cv2.resize(image, (width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # The bigger the kernel, the more the white region increases.
    # If the resizing step was ignored, then the kernel will have to be bigger
    # than the one given here.
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('image', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        xmin, ymin, width, height = cv2.boundingRect(cnt)
        extent = area / (width * height)

        # filter non-rectangular objects and small objects
        if (extent > np.pi / 4) and (area > 100):
            bboxes = [xmin, ymin, xmin + width, ymin + height]
    print("bboxes:", bboxes, "\n")
    if bboxes == []:
        return [[0,0],[0,0],[0,0],[0,0]]
    return bboxes

def get_qr_corners_from_colorseg_image(image):
    image = np.array(image)
    orig_image = copy.deepcopy(image)
    # plt.title("image")
    # plt.imshow(image)
    # plt.show()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (0, 200, 200) #(0, 200, 0)
    dark_color = (100, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    inverted_img = util.invert(imgGray)
    inverted_img = np.uint8(inverted_img)
    inverted_img = 255 - inverted_img

    # # Set up the SimpleBlobdetector with default parameters.
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(inverted_img)
    print("keypoints:", len(keypoints))

    if keypoints == []:
        return [[0, 0], [0, 0], [0,0], [0,0]], None
    else:
        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]),
                                              (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.title("im_with_keypoints")
        plt.imshow(im_with_keypoints)
        plt.pause(0.01)
        # kpts = cv2.KeyPoint_convert(keypoints)
        bboxes = [[(int(keypoint.pt[0] - keypoint.size/2), int(keypoint.pt[1] - keypoint.size/2)),
                   (int(keypoint.pt[0] + keypoint.size / 2), int(keypoint.pt[1] - keypoint.size / 2)),
                   (int(keypoint.pt[0] - keypoint.size / 2), int(keypoint.pt[1] + keypoint.size / 2)),
                   (int(keypoint.pt[0] + keypoint.size/2), int(keypoint.pt[1] + keypoint.size/2))] for keypoint in keypoints]
        print("bboxes:", bboxes)
        color = (255, 0, 0)
        thickness = 1
        # boxedimg = cv2.rectangle(image, start_point, end_point, color, thickness)
        boxedimg = cv2.rectangle(orig_image, bboxes[0][0], bboxes[0][1], color, thickness)
        plt.title("bboxed img")
        plt.imshow(boxedimg)
        plt.pause(0.01)
        print("point sizes:", [keypoint.size for keypoint in keypoints], "\n")
        return bboxes, boxedimg

def setup_beamng(vehicle_model='etk800', camera_pos=(-0.5, 0.38, 1.3), camera_direction=(0, 1.0, 0), pitch_euler=0.0):
    global base_filename, default_color, default_scenario,default_spawnpoint, steps_per_sec
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0

    # setup DNN model + weights
    sm = DAVE2Model()
    # steering_model = Model().define_model_BeamNG("BeamNGmodel-racetracksteering8.h5")
    # throttle_model = Model().define_model_BeamNG("BeamNGmodel-racetrackthrottle8.h5")
    dual_model = sm.define_dual_model_BeamNG()
    # dual_model = sm.load_weights("BeamNGmodel-racetrackdualcomparison10K.h5")
    # dual_model = sm.define_multi_input_model_BeamNG()
    # dual_model = sm.load_weights("BeamNGmodel-racetrackdual-comparison10K-PIDcontrolset-4.h5")
    # dual_model = sm.load_weights("BeamNGmodel-racetrackdual-comparison40K-PIDcontrolset-1.h5")
    # BeamNGmodel-racetrack-multiinput-dualoutput-comparison10K-PIDcontrolset-1.h5
    # BeamNGmodel-racetrackdual-comparison100K-PIDcontrolset-2
    dual_model = sm.load_weights("BeamNGmodel-racetrackdual-comparison100K-PIDcontrolset-2.h5")
    # dual_model = sm.load_weights("BeamNGmodel-racetrack-multiinput-dualoutput-comparison103K-PIDcontrolset-1.h5")

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
    # Compile the scenario and place it in BeamNG's map folder
    scenario.make(beamng)

    # Start BeamNG and enter the main loop
    bng = beamng.open(launch=True)
    #bng.hide_hud()
    bng.set_deterministic()  # Set simulator to be deterministic
    bng.set_steps_per_second(steps_per_sec)  # With 60hz temporal resolution
    bng.load_scenario(scenario)
    bng.start_scenario()
    # Put simulator in pause awaiting further inputs
    bng.pause()
    assert vehicle.skt
    # bng.resume()
    # find_width_of_road(bng)
    return vehicle, bng, spawn, dual_model

def run_scenario(vehicle, bng, dual_model, vehicle_model='etk800', camera_pos=(-0.5, 0.38, 1.3), camera_direction=(0, 1.0, 0), pitch_euler=0.0, run_number=0):
    global base_filename, default_color, default_scenario,default_spawnpoint, steps_per_sec
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0
    bng.restart_scenario()

    # perturb vehicle
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    image = sensors['front_cam']['colour'].convert('RGB')
    plt.title("first image cam_pos={} cam_dir={}\npitch_euler={}".format(camera_pos, camera_direction, pitch_euler))
    plt.imshow(image)
    plt.pause(0.01)
    pitch = vehicle.state['pitch'][0]
    roll = vehicle.state['roll'][0]
    z = vehicle.state['pos'][2]
    spawn = spawn_point(default_scenario, default_spawnpoint)

    # print("VEHICLE BOUNDING BOX:{}".format(vehicle.get_bbox()))
    prev_error = setpoint; damage_prev = None; runtime = 0.0
    kphs = []; traj = []; pitches = []; rolls = []; steering_inputs = []; throttle_inputs = []; timestamps = []
    damage = None; overall_damage = 0.0
    final_img = None
    total_loops = 0; total_imgs = 0; total_predictions = 0
    start_time = sensors['timer']['time']
    # filename = "{}{}.txt".format(base_filename, run_number)
    filename = "{}coordinates.txt".format(base_filename)
    imagecount = 0
    with open(filename, "w") as f:
        # f.write("FILENAME,BOUNDING_BOXES\n")
        while overall_damage <= 0:
            # collect images
            sensors = bng.poll_sensors(vehicle)
            # print(sensors['front_cam'].keys())
            # exit(0)
            image = sensors['front_cam']['colour'].convert('RGB')
            print("filename", filename)
            # plt.title("original image ")
            # plt.plot(image)
            # plt.pause(0.01)
            colorseg_img = sensors['front_cam']['annotation'].convert('RGB')
            depth_img = sensors['front_cam']['depth'].convert('RGB')
            qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
            if bbox_img is not None:
                # save the image
                full_filename = "{}{}.jpg".format(base_filename, imagecount)
                print("image full_filename:", full_filename)
                plt.title("semantic segmentation")
                plt.imshow(colorseg_img)
                plt.pause(0.01)
                image.save(full_filename)

                # f.write("{},{}\n".format(relative_filename, str(qr_corners).replace(",","")))
                new_line = "\t{}\t{}\n".format(imagecount, str(qr_corners).replace(",", "").replace("]", "").replace("[", " ")).replace("(", "").replace(")", "")
                print(new_line)
                f.write(new_line)

                total_imgs += 1
                imagecount += 1

            img = sm.process_image(np.asarray(image))
            wheelspeed = sensors['electrics']['wheelspeed']
            kph = ms_to_kph(wheelspeed)
            dual_prediction = dual_model.predict(x=[img, np.array([kph])])
            dt = (sensors['timer']['time'] - start_time) - runtime
            runtime = sensors['timer']['time'] - start_time

            # control params
            brake = 0
            steering = float(dual_prediction[0][0])
            # throttle = float(dual_prediction[0][1])
            total_predictions += 1
            if abs(steering) > 0.2:
                setpoint = 20
            else:
                setpoint = 40

            throttle = throttle_PID(kph, dt)

            vehicle.control(throttle=throttle, steering=steering, brake=brake)

            steering_inputs.append(steering)
            throttle_inputs.append(throttle)
            timestamps.append(runtime)

            damage = sensors['damage']
            overall_damage = damage["damage"]
            new_damage = diff_damage(damage, damage_prev)
            damage_prev = damage
            vehicle.update_vehicle()
            traj.append(vehicle.state['pos'])
            pitches.append(vehicle.state['pitch'][0])
            rolls.append(vehicle.state['roll'][0])

            kphs.append(ms_to_kph(wheelspeed))
            total_loops += 1
            final_img = image

            if new_damage > 0.0:
                print("New damage={}, exiting...".format(new_damage))
                break
            if runtime > 5:
                print("Exited after 5 minutes successful runtime")
                break
            if distance(spawn['pos'], vehicle.state['pos']) < 5 and runtime > 10:
                print("Completed one lap, exiting...")
                break
            outside_track = has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng)
            if outside_track:
                print("Left track, exiting...")
                break

            bng.step(1, wait=True)

            # del image
    #     print("runtime:{}".format(round(runtime, 2)))
    # print("time to crash:{}".format(round(runtime, 2)))
    # bng.close()
    # avg_kph = float(sum(kphs)) / len(kphs)
    plt.imshow(final_img)
    plt.savefig("Run-{}-finalimg.png".format(run_number))
    plt.pause(0.01)

    results = "Total loops: {} \ntotal images: {} \ntotal predictions: {} \nexpected predictions ={}*{}={}".format(total_loops, total_imgs, total_predictions, round(runtime,3), steps_per_sec, round(runtime*steps_per_sec,3))
    print(results)
    deviation = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime,3), 'damage': damage, 'kphs':kphs, 'traj':traj, 'pitch': round(pitch,3),
               'roll':round(roll,3), "z":round(z,3), 'final_img':final_img,
               'camera_direction':camera_direction, 'camera_position':camera_pos,
               'deviation':deviation
               }
    return results

def run_scenario_old(vehicle_model='etk800', camera_pos=(-0.5, 0.38, 1.3), camera_direction=(0, 1.0, 0), pitch_euler=0.0, run_number=0):
    global base_filename, default_color, default_scenario,default_spawnpoint, steps_per_sec
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0

    # setup DNN model + weights
    sm = DAVE2Model()
    # steering_model = Model().define_model_BeamNG("BeamNGmodel-racetracksteering8.h5")
    # throttle_model = Model().define_model_BeamNG("BeamNGmodel-racetrackthrottle8.h5")
    dual_model = sm.define_dual_model_BeamNG()
    # dual_model = sm.load_weights("BeamNGmodel-racetrackdualcomparison10K.h5")
    # dual_model = sm.define_multi_input_model_BeamNG()
    # dual_model = sm.load_weights("BeamNGmodel-racetrackdual-comparison10K-PIDcontrolset-4.h5")
    # dual_model = sm.load_weights("BeamNGmodel-racetrackdual-comparison40K-PIDcontrolset-1.h5")
    # BeamNGmodel-racetrack-multiinput-dualoutput-comparison10K-PIDcontrolset-1.h5
    # BeamNGmodel-racetrackdual-comparison100K-PIDcontrolset-2
    dual_model = sm.load_weights("BeamNGmodel-racetrackdual-comparison100K-PIDcontrolset-2.h5")
    # dual_model = sm.load_weights("BeamNGmodel-racetrack-multiinput-dualoutput-comparison103K-PIDcontrolset-1.h5")

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
    # Compile the scenario and place it in BeamNG's map folder
    scenario.make(beamng)

    # Start BeamNG and enter the main loop
    bng = beamng.open(launch=True)
    #bng.hide_hud()
    bng.set_deterministic()  # Set simulator to be deterministic
    bng.set_steps_per_second(steps_per_sec)  # With 60hz temporal resolution
    bng.load_scenario(scenario)
    bng.start_scenario()
    # Put simulator in pause awaiting further inputs
    bng.pause()
    assert vehicle.skt
    # bng.resume()

    # perturb vehicle
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    image = sensors['front_cam']['colour'].convert('RGB')
    plt.title("first image cam_pos={} cam_dir={}\npitch_euler={}".format(camera_pos, camera_direction, pitch_euler))
    plt.imshow(image)
    plt.pause(0.01)
    pitch = vehicle.state['pitch'][0]
    roll = vehicle.state['roll'][0]
    z = vehicle.state['pos'][2]

    print("VEHICLE BOUNDING BOX:{}".format(vehicle.get_bbox()))
    prev_error = setpoint; damage_prev = None; runtime = 0.0
    kphs = []; traj = []; pitches = []; rolls = []; steering_inputs = []; throttle_inputs = []; timestamps = []
    damage = None
    final_img = None
    overall_damage = 0.0
    total_loops = 0; total_imgs = 0; total_predictions = 0
    while overall_damage <= 0:
        # collect images
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_cam']['colour'].convert('RGB')

        total_imgs += 1
        img = sm.process_image(np.asarray(image))
        wheelspeed = sensors['electrics']['wheelspeed']
        kph = ms_to_kph(wheelspeed)
        dual_prediction = dual_model.predict(x=[img, np.array([kph])])
        # steering_prediction = steering_model.predict(img)
        # throttle_prediction = throttle_model.predict(img)
        dt = sensors['timer']['time'] - runtime
        runtime = sensors['timer']['time']

        # control params

        brake = 0
        # steering = float(steering_prediction[0][0]) #random.uniform(-1.0, 1.0)
        # throttle = float(throttle_prediction[0][0])
        steering = float(dual_prediction[0][0]) #random.uniform(-1.0, 1.0)
        throttle = float(dual_prediction[0][1])
        total_predictions += 1
        if abs(steering) > 0.2:
            setpoint = 20
        else:
            setpoint = 40
        # if runtime < 10:
        throttle = throttle_PID(kph, dt)
        #     if throttle > 1:
        #         throttle = 1
            # if setpoint < kph:
            #     brake = 0.0 #throttle / 10000.0
            #     throttle = 0.0
        vehicle.control(throttle=throttle, steering=steering, brake=brake)

        steering_inputs.append(steering)
        throttle_inputs.append(throttle)
        timestamps.append(runtime)

        steering_state = sensors['electrics']['steering']
        steering_input = sensors['electrics']['steering_input']
        avg_wheel_av = sensors['electrics']['avg_wheel_av']

        damage = sensors['damage']
        overall_damage = damage["damage"]
        new_damage = diff_damage(damage, damage_prev)
        damage_prev = damage
        vehicle.update_vehicle()
        traj.append(vehicle.state['pos'])
        pitches.append(vehicle.state['pitch'][0])
        rolls.append(vehicle.state['roll'][0])

        kphs.append(ms_to_kph(wheelspeed))
        total_loops += 1
        final_img = image

        if new_damage > 0.0:
            break
        bng.step(1, wait=True)

        # if runtime > 300:
        #     print("Exited after 5 minutes successful runtime")
        #     break

        if distance(spawn['pos'], vehicle.state['pos']) < 5 and sensors['timer']['time'] > 10:
            reached_start = True
            break

    #     print("runtime:{}".format(round(runtime, 2)))
    # print("time to crash:{}".format(round(runtime, 2)))
    bng.close()
    # avg_kph = float(sum(kphs)) / len(kphs)
    plt.imshow(final_img)
    plt.savefig("Run-{}-finalimg.png".format(run_number))
    plt.pause(0.01)
    plot_input(timestamps, steering_inputs, "Steering", run_number=run_number)
    plot_input(timestamps, throttle_inputs, "Throttle", run_number=run_number)
    plot_input(timestamps, kphs, "KPHs", run_number=run_number)
    print("Number of steering_inputs:", len(steering_inputs))
    print("Number of throttle inputs:", len(throttle_inputs))
    results = "Total loops: {} \ntotal images: {} \ntotal predictions: {} \nexpected predictions ={}*{}={}".format(total_loops, total_imgs, total_predictions, round(runtime,3), steps_per_sec, round(runtime*steps_per_sec,3))
    print(results)
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

def add_qr_cubes(scenario):
    barrier_locations = []
    with open('qr_box_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            # rot_quat = turn_90(rot_quat)
            # barrier_locations.append({'pos':pos, 'rot_quat':rot_quat})
            # add barrier to scenario
            cube = ScenarioObject(oid='qrbox_{}'.format(i),
                              name='qrbox2',
                              otype='BeamNGVehicle',
                              pos=pos,
                              rot=None,
                              rot_quat=(0,0,0,1),
                              scale=(500,500,500),
                              JBeam = 'qrbox2',
                              datablock="default_vehicle"
                              )
            # mesh = ProceduralCube(pos=pos, rot=None, size=10, name='cube{}'.format(i), rot_quat=rot_quat, material=None)
            # scenario.add_procedural_mesh(mesh)
            # H:\BeamNG.research.v1.7.0.1clean\levels\west_coast_usa\art\shapes\objects/solarpanel.dae
            ramp = StaticObject(name='cube{}'.format(i), pos=pos, rot=None, rot_quat=(0,0,0,1), scale=(1, 1, 1),
                                shape='levels/Industrial/art/shapes/misc/container_01_a.dae')
            # ramp = StaticObject(name='cube{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
            #                     shape='levels/west_coast_usa/art/shapes/objects/solarpanel.dae') #barrierfence_plain.dae') #billboard_structure1.dae') #solarpanel.dae')

            scenario.add_object(cube)

def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))

def main():
    global base_filename, default_color, default_scenario, setpoint, integral
    global prev_error, centerline
    camera_zs = [1.3] #[0.87, 1.07, 1.3, 1.51, 1.73]
    camera_pitches = [] #[-15, -5, 0, 5, 15] #[-30, -15, 0, 15, 30] # in euler angles, convert to rotation vectors
    # intake_ai_lap_poses()
    # create_ai_line_from_road_with_interpolation()
    for z in camera_zs:
        camera_pos = (-0.5, 0.38, z)
        camera_dir = (0, 1.0, 0)
        vehicle, bng, spawn, model = setup_beamng(vehicle_model='hopper', camera_pos=camera_pos,
                                                    camera_direction=camera_dir)
        create_ai_line_from_road(spawn, bng)
        for i in range(1):
            print("\nRun {} for camera z={}".format(i, z))
            results = run_scenario(vehicle, bng, model, vehicle_model='hopper', camera_pos=camera_pos, camera_direction=camera_dir, run_number=i)
            results['distance'] = get_distance_traveled(results['traj'])
            sample_entry = '{},{},{},{},{},{},{}'.format(str(camera_dir).replace(",", ""), 0,
                                                      str(camera_pos).replace(",", ""),
                                                      results['runtime'], results['distance'], results['deviation']
                                                      )
            plot_trajectory(results['traj'], sample_entry, run_number=i)
        bng.close()
    for pitch in camera_pitches:
        camera_pos = (-0.5, 0.38, 1.3)
        pitch_rotvec = R.from_euler('xyz', (0, 0, pitch), degrees=True)
        camera_dir = pitch_rotvec.as_rotvec() + (0, 1, 0)  # (0,1,-0.25) #- (0,  1.28407346, 0)
        vehicle, bng, sm, dual_model = setup_beamng(vehicle_model='hopper', camera_pos=camera_pos,
                                                    camera_direction=camera_dir)
        for i in range(0):
            print("\nRun {} for camera pitch={}".format(i, pitch))
            results = run_scenario(vehicle, bng, sm, dual_model, vehicle_model='hopper', camera_pos=camera_pos, camera_direction=camera_dir, pitch_euler=pitch, run_number=i)
            results['distance'] = get_distance_traveled(results['traj'])
            score = calculate_score(results)
            plot_trajectory(results['traj'], sample_entry, run_number=i)
        bng.close()

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    main()
