# meriel@DESKTOP-FFUNBFQ:/mnt/c/Users/merie$ cd /mnt/h/GitHub/superdeepbillboard/superdeepbillboard/simulation/
# meriel@DESKTOP-FFUNBFQ:/mnt/h/GitHub/superdeepbillboard/superdeepbillboard/simulation$ /mnt/c/Users/merie/.virtualenvs/BeamNGpy-master-EahKJuG0/Scripts/python.exe deepbillboard-collect-perturbed-trace-metas.py

import argparse
#https://docs.python.org/3/library/argparse.html
parser = argparse.ArgumentParser(description='Process paths')
parser.add_argument('path2src', metavar='N', type=str, help='path to source parent dirs')
parser.add_argument('road_id', metavar='N', type=str, help='road identifier in BeamNG')
args = parser.parse_args()
print(args)

import warnings
import numpy as np
from matplotlib import pyplot as plt
import logging, random, string, time, copy, os, sys, shutil

sys.path.append(f'{args.path2src}/GitHub/DAVE2-Keras')
sys.path.append(f'{args.path2src}/GitHub/superdeepbillboard')
sys.path.append(f'{args.path2src}/GitHub/BeamNGpy')
sys.path.append(f'{args.path2src}/GitHub/BeamNGpy/src/')
print(sys.path)

from deepbillboard import DeepBillboard, SuperDeepBillboard
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy import ProceduralCube #,ProceduralCylinder, ProceduralCone, ProceduralBump, ProceduralRing
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

import statistics, math
from scipy.spatial.transform import Rotation as R
from ast import literal_eval
from scipy import interpolate
import torch
import cv2
from skimage import util
from PIL import Image
from sklearn.metrics import mean_squared_error
import kornia
from torchvision.utils import save_image
import pandas as pd
import pickle
from shapely.geometry import Polygon

from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from functools import wraps
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

# globals
default_color = 'White'  # 'Red'
default_scenario = "industrial" #'driver_training'  # 'automation_test_track'
default_spawnpoint = "curve2" #"straight1" #"approachingfork"  # "curve1" #
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40.0 #40  #53.3 #https://en.wikipedia.org/wiki/Speed_limits_by_country
setpoint = overall_throttle_setpoint
lanewidth = 3.75  # 2.25
centerline, centerline_interpolated = [], []
roadmiddle, roadleft, roadright = [], [], []
expected_trajectory = []
steps_per_sec = 15  # prev: 60, 100
# collection_hash = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
newdir, new_results_dir = '', ''
qr_positions = []
unperturbed_traj, unperturbed_steer = [], []


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

def spawn_point(scenario_locale, spawn_point='default'):
    global lanewidth
    if scenario_locale == 'industrial':
        # racetrack sequence starting points
        if spawn_point == "curve1":
            # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.0022243109997362,0.0038272379897535,0.92039221525192,-0.39097133278847)}
            return {'pos': (210.314, -44.7132, 42.7758), 'rot': None,
                    'rot_quat': (0.0020199827849865, 0.0049774856306612, 0.92020887136459, -0.3913908302784)}
        elif spawn_point == "curve2":
            return {'pos': (319.606, -92.2611, 43.7064), 'rot': None,
                    'rot_quat': (0.0078629404306412, 0.012614111416042, -0.37881037592888, 0.92535495758057)}
        # elif spawn_point == "curve2":
        #     return {'pos':(323.432,-92.7588,43.6475), 'rot': None, 'rot_quat':(0.008327,0.0137599,-0.36539,0.930714)}
        #     # 172.713|E|libbeamng.lua.V.updateGFX|Object position: vec3(327.801,-100.41,43.9318)
        #     # 172.716|E|libbeamng.lua.V.updateGFX|Object rotation (quat): quat(0.0087151182815433,0.020582119002938,-0.36003017425537,0.93267297744751)
        #     return {'pos': (331.169, -104.166, 44.142), 'rot': None, 'rot_quat': (0.009578, 0.033658, -0.359433, 0.93251)}
        elif spawn_point == "curve3":
            return {'pos': (270.271, -206.988, 43.9982), 'rot': None,
                    'rot_quat': (-0.0071861492469907, 0.0055168853141367, 0.47670912742615, 0.87901443243027)}
        elif spawn_point == "curve4":  # beginning of big curve right before starting gate
            # 50kph
            # return {'pos': (171.522, -109.746, 44.8168), 'rot': None,
            #         'rot_quat': (0.02015926875174, -0.023340219631791, 0.18135184049606, 0.982934653759)}
            # return {'pos': (169.177, -115.208, 44.4244), 'rot': None,
            #         'rot_quat': (0.022967331111431, -0.019803000614047, 0.19603483378887, 0.98012799024582)}
            # 40kph
            return {'pos': (166.291, -124.855, 43.7002), 'rot': None,
                    'rot_quat': (-0.023691736161709, -0.028266951441765, 0.14190191030502, 0.989193379879)}
        elif spawn_point == "straight1":
            # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.0022243109997362,0.0038272379897535,0.92039221525192,-0.39097133278847)}
            # 130 steps
            # return {'pos': (252.028,-24.7376,42.814), 'rot': None,'rot_quat': (-0.044106796383858,0.05715386942029,-0.49562504887581,0.8655309677124)}
            # 50 steps
            return {'pos': (257.414, -27.9716, 42.8266), 'rot': None,
                    'rot_quat': (-0.032358665019274, 0.05354256555438, -0.45097458362579, 0.89034152030945)}
            # 4 steps
            return {'pos': (265.087, -33.7904, 42.805), 'rot': None,
                    'rot_quat': (-0.022659547626972, 0.023112617433071, -0.42281490564346, 0.90563786029816)}
        # elif spawn_point == "straight4":
    elif scenario_locale == 'driver_training':
        if spawn_point == "north":
            return {'pos':(-195.047, 253.654, 53.019), 'rot': None, 'rot_quat':(-0.006, -0.006, -0.272, 0.962)}
        elif spawn_point == "west":
            return {'pos': (-394.541, 69.052, 51.2327), 'rot': None, 'rot_quat': (-0.0124, 0.0061, -0.318, 0.948)}
        elif spawn_point == "default":
            return {'pos':(60.6395, 70.8329, 38.3048), 'rot': None, 'rot_quat':(0.015, 0.006, 0.884, 0.467)}
            #return {'pos': (32.3209, 89.8991, 39.135), 'rot': None, 'rot_quat': (0.0154, -0.007, 0.794, 0.607)}
        elif spawn_point == "misshapenstraight":
            return {'pos': (-111.879, 174.348, 50.5944), 'rot': None, 'rot_quat': (-0.012497862800956, -0.0070292484015226, -0.57099658250809, 0.82082730531693)}
        elif spawn_point == "approachingfork":
            return {'pos': (48.5345, 188.014, 48.2153), 'rot': None, 'rot_quat': (-0.013060956262052, -0.019843459129333, 0.80683600902557, 0.5902978181839)}
            # return {'pos': (18.7422,196.851,49.1215), 'rot': None,'rot_quat': (-0.03395925834775,-0.019455011934042,0.79710978269577,0.6025647521019)}
            # return {'pos': (18.7422, 196.851, 49.1215), 'rot': None,'rot_quat': (-0.03395925834775, -0.019455011934042, 0.79710978269577, 0.6025647521019)}

def get_sequence_setup(sequence):
    df = pd.read_csv("posefiles/sequence-setup.txt", sep="\s+")
    keys = df.keys()
    index = df.index[df['sequence'] == sequence].tolist()[0]
    vals = {key: df.at[index, key] for key in keys}
    print(vals)
    return vals

def setup_sensors(vehicle, pos=(-0.5, 0.38, 1.3), direction=(0, 1.0, 0)):
    fov = 50  # 60 works for full lap #63 breaks on hairpin turn
    resolution = (240, 135)  # (400,225) # (240, 135)  # (200, 150) (320, 180) #(1280,960) #(512, 512)
    front_camera = Camera(pos, direction, fov, resolution,
                          colour=True, depth=True, annotation=True)
    birdseye_cam = Camera((-0.5, 0.38, 100), (0,1,-1), fov, resolution, colour=True, depth=True, annotation=True)

    gforces = GForces()
    electrics = Electrics()
    damage = Damage()
    timer = Timer()

    # Attach them
    vehicle.attach_sensor('front_cam', front_camera)
    vehicle.attach_sensor('overhead', birdseye_cam)
    vehicle.attach_sensor('gforces', gforces)
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('damage', damage)
    vehicle.attach_sensor('timer', timer)
    return vehicle

def add_barriers(scenario):
    with open('posefiles/industrial_racetrack_barrier_locations.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            # turn barrier 90 degrees
            r = R.from_quat(list(rot_quat))
            r = r.as_euler('xyz', degrees=True)
            r[2] = r[2] + 90
            r = R.from_euler('xyz', r, degrees=True)
            rot_quat = tuple(r.as_quat())
            barrier = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
            # barrier.type="BeamNGVehicle"
            scenario.add_object(barrier)

def add_qr_cubes(scenario):
    global qr_positions
    qr_positions = []
    with open('posefiles/qr_box_locations-curve2.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
            box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
                                 rot_quat=rot_quat, scale=(5,5,5), JBeam='qrbox2', datablock="default_vehicle")
            scenario.add_object(box)
        if default_scenario == "industrial" and (default_spawnpoint == "curve4"):
            cube = ProceduralCube(name='cube_platform',
                                  # pos=(145.214,-160.72,43.7269),
                                  pos=(150, -170, 44),
                                  rot=None,
                                  rot_quat=(0, 0, 0, 1),
                                  size=(2, 6, 0.5))
            scenario.add_procedural_mesh(cube)
        elif default_scenario == "industrial" and (default_spawnpoint == "curve2"):
            # turn barrier 90 degrees
            r = R.from_quat(list((-0.007496957767951349, 0.010966399078630984, 0.3849858989069824, 0.9228268479830095)))
            r = r.as_euler('xyz', degrees=True)
            r[1] = 0
            r[2] = r[2]+90
            r = R.from_euler('xyz', r, degrees=True)
            rot_quat = tuple(r.as_quat())
            print(f"{rot_quat=}")
            print(f"{rot_quat=}")
            print(f"{rot_quat=}")
            print(f"{rot_quat=}")
            cube = ProceduralCube(name='cube_platform',
                                  pos=(352.246,-127.169,45),
                                  rot=None,
                                  rot_quat=(-0.001026031748168209, -0.002494647455262238, 0.9248281562457762, 0.3803756109531261),
                                  size=(3, 6, 3))
            scenario.add_procedural_mesh(cube)
        elif default_scenario == "driver_training" and default_spawnpoint == "approachingfork":
            cube = ProceduralCube(name='cube_platform',
                                  pos=(-20.3113, 218.448, 50.043),
                                  rot=None,
                                  rot_quat=(-0.022064134478569,-0.022462423890829,0.82797580957413,0.55987912416458),
                                  size=(4, 8, 0.5))
            scenario.add_procedural_mesh(cube)

def get_outcomes(results):
    outcomes_counts = {"D":0, "LT":0, "R2NT":0, "2FAR":0}
    # print(results.keys())
    total = float(len(results["testruns_outcomes"]) - 1)
    for outcome in results['testruns_outcomes']:
        if "D=" in outcome:
            outcomes_counts["D"] += 1
        elif "R2NT" in outcome:
            outcomes_counts["R2NT"] +=1
        elif "LT" in outcome:
            outcomes_counts["LT"] += 1
        elif "2FAR" in outcome:
            outcomes_counts["2FAR"] += 1
    return outcomes_counts

def ms_to_kph(wheelspeed):
    return wheelspeed * 3.6

def throttle_PID(kph, dt):
    global integral, prev_error, setpoint
    # kp = 0.001; ki = 0.00001; kd = 0.0001
    # kp = .3; ki = 0.01; kd = 0.1
    # kp = 0.15; ki = 0.0001; kd = 0.008 # worked well but only got to 39kph
    kp = 0.19
    ki = 0.0001
    kd = 0.008
    error = setpoint - kph
    if dt > 0:
        deriv = (error - prev_error) / dt
    else:
        deriv = 0
    integral = integral + error * dt
    w = kp * error + ki * integral + kd * deriv
    prev_error = error
    return w

def plot_deviation(trajectories, unperturbed_traj, model, centerline, left, right, outcomes,
                   xlim=[100, 350], ylim=[-260, 0], resultsdir="images"):
    global qr_positions
    x = [point[0] for point in unperturbed_traj]
    y = [point[1] for point in unperturbed_traj]
    plt.plot(x, y, label="Unpert", linewidth=10)
    x, y = [], []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    x.append(centerline[0][0])
    y.append(centerline[0][1])
    plt.plot(x, y, 'k')
    x, y = [], []
    for point in left:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, 'k')
    x, y = [], []
    for point in right:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, 'k')
    x, y = [], []
    for i, t in enumerate(zip(trajectories, outcomes)):
        x = [point[0] for point in t[0]]
        y = [point[1] for point in t[0]]
        if i == 0 and 'sdbb' in model:
            plt.plot(x, y, label="Pert. Run {} ({})".format(i, t[1]), linewidth=5)
        else:
            plt.plot(x, y, label="Pert. Run {} ({})".format(i, t[1]), alpha=0.75)
        i += 1
    plt.plot([p[0][0] for p in qr_positions], [p[0][1] for p in qr_positions], 'r', linewidth=5)
    failures = 0
    for o in outcomes:
        if o == "LT" or "D" in o:
            failures += 1
    plt.title('Trajectories with {} failures={}'.format(model, failures), fontdict={'fontsize': 10})
    # plt.legend()
    if default_spawnpoint == "straight1" and "ZOOMED" not in model:
        plt.xlim([245, 335])
        plt.ylim([-123, -20])
    elif default_spawnpoint == "straight1" and "ZOOMED" in model:
        plt.xlim([265, 300])
        plt.ylim([-60, -30])
    elif default_spawnpoint == "curve1":
        plt.xlim([265, 300])
        plt.ylim([-60, -30])
    elif default_spawnpoint == "curve2":
        plt.xlim([325, 345])
        plt.ylim([-120, -100])
    elif default_spawnpoint == "curve4":
        plt.xlim([180, 120])
        plt.ylim([-120, -180])
    elif default_scenario == "driver_training" and default_spawnpoint == "approachingfork":
        plt.xlim([-50, 55])
        plt.ylim([150, 255])
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    plt.savefig("{}/{}-{}.jpg".format(resultsdir, model.replace("\n", "-"), randstr))
    plt.close("all")
    del x, y


def plot_steering(unperturbed_all_ys, pertrun_all_ys, testruns_all_ys, title="", resultsdir="results"):
    for i, ys in enumerate(testruns_all_ys):
        plt.plot(range(len(ys)), ys)
    # plot these last so they're easier to see
    plt.plot(range(len(unperturbed_all_ys)), unperturbed_all_ys, label="unpert.", linewidth=6)
    # handle DBB case
    if pertrun_all_ys is not None:
        plt.plot(range(len(pertrun_all_ys)), pertrun_all_ys, label="intended pert.", linewidth=5)
    plt.legend()
    plt.title(f"Steering inputs for\n{title}")
    plt.savefig("{}/steering-{}.jpg".format(resultsdir, title))
    plt.close("all")


def plot_errors(errors, filename="images/errors.png"):
    plt.title("Errors")
    for ei, e in enumerate(errors):
        plt.plot(range(len(e)), e, label=f"Error {ei}")
    plt.savefig("{}".format(filename))
    plt.close("all")
    plt.title("Error Distributions per Run")
    avgs = []
    for ei, e in enumerate(errors):
        plt.scatter(np.ones((len(e)))*ei, e, s=5)
        avgs.append(float(sum(e)) / len(e))
    plt.plot(range(len(avgs)), avgs)
    plt.savefig("{}".format(filename.replace(".png", "-distribution.png")))
    plt.close("all")

@ignore_warnings
def write_results(training_file, results, all_trajs, unperturbed_traj,
                                  modelname, technique, direction, lossname, bbsize, its, nl):
    results["all_trajs"] = all_trajs
    results["unperturbed_traj"] = unperturbed_traj
    results["modelname"] = modelname
    results["technique"] = technique
    results["direction"] = direction
    results["lossname"] = lossname
    results["bbsize"] = bbsize
    results["its"] = its
    results["nl"] = nl
    with open(training_file, "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


@ignore_warnings
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
    # signed parallel distance components (rowwise dot products of 2D vectors)
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])
    # perpendicular distance component (rowwise cross products of 2D vectors)
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)

# return distance between two 3d points
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def dist_from_line(centerline, point):
    a = [[x[0], x[1]] for x in centerline[:-1]]
    b = [[x[0], x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist

def calc_deviation_from_center(centerline, traj):
    dists = []
    for point in traj:
        dist = dist_from_line(centerline, point)
        dists.append(min(dist))
    avg_dist = sum(dists) / len(dists)
    stddev = statistics.stdev(dists)
    return stddev, dists, avg_dist

def intake_lap_file(filename="posefiles/DAVE2v3-lap-trajectory.txt"):
    global expected_trajectory
    expected_trajectory = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = literal_eval(line)
            expected_trajectory.append(line)

def plot_racetrack_roads(roads, bng):
    global default_scenario, default_spawnpoint
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    print(f"{len(roads)=}")
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp = []
        y_temp = []
        dont_add = False
        xy_def = [edge['middle'][:2] for edge in road_edges]
        dists = [distance2D(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
        # if sum(dists) > 500 and (road != "9096" or road != "9206"):
        s = sum(dists)
        if (s < 150):
            continue
        for edge in road_edges:
            # if edge['middle'][1] > -200: #< 0:
            #     dont_add=True
            #     break
            # if edge['middle'][0] < 100:
            #     dont_add = True
            #     break
            # if edge['middle'][1] < -300 or edge['middle'][1] > 0:
            #     dont_add = True
            #     break
            if not dont_add:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
        if not dont_add:
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
    plt.legend()
    plt.title("{} {}".format(default_scenario, default_spawnpoint))
    plt.show()
    plt.pause(0.001)

def road_analysis(bng):
    global centerline, roadleft, roadright, roadmiddle
    global default_scenario, default_spawnpoint
    plot_racetrack_roads(bng.get_roads(), bng)
    # get relevant road
    edges = []
    adjustment_factor = 4.0
    if default_scenario == "industrial" and default_spawnpoint == "racetrackstartinggate":
        edges = bng.get_road_edges('7982')
    elif default_scenario == "industrial" and (default_spawnpoint == "straight1" or default_spawnpoint == "curve2" or default_spawnpoint == "curve1"  or default_spawnpoint == "curve4"):
        edges = bng.get_road_edges(args.road_id)
        adjustment_factor = 10.0
    elif default_scenario == "industrial" and default_spawnpoint == "driftcourse2":
        edges = bng.get_road_edges('7987')
    elif default_scenario == "hirochi_raceway" and default_spawnpoint == "startingline":
        edges = bng.get_road_edges('9096')
        edges.extend(bng.get_road_edges('9206'))
        # edges = bng.get_road_edges('9206')
    elif default_scenario == "utah" and default_spawnpoint == "westhighway":
        edges = bng.get_road_edges('15145')
        # edges.extend(bng.get_road_edges('15162'))
        edges.extend(bng.get_road_edges('15154'))
        edges.extend(bng.get_road_edges('15810'))
        edges.extend(bng.get_road_edges('16513'))
        adjustment_factor = 1.0
    elif default_scenario == "utah" and default_spawnpoint == "westhighway2":
        edges = bng.get_road_edges('15810')
        # edges.extend(bng.get_road_edges('15810'))
        edges.extend(bng.get_road_edges('16513'))
        # edges.extend(bng.get_road_edges('15143'))
        # edges = bng.get_road_edges('9206')
        adjustment_factor = 1.0
    elif default_scenario == "utah" and default_spawnpoint == "undef":
        edges = bng.get_road_edges('15852')
        edges.extend(bng.get_road_edges('14904'))
        edges.extend(bng.get_road_edges('15316'))
        adjustment_factor = 1.0
    elif default_scenario == "driver_training" and default_spawnpoint == "approachingfork":
        edges = bng.get_road_edges("7719")
        edges.reverse()
        adjustment_factor = -0.3
        # adjustment_factor = -0.01
        for i in range(len(edges)):
            # edges[i]['left'] = np.array(edges[i]['middle']) + (np.array(edges[i]['left']) - np.array(edges[i]['middle']))/ -0.1
            edges[i]['right'] = np.array(edges[i]['middle']) + (np.array(edges[i]['right']) - np.array(edges[i]['middle']))/ 0.1
        # edges = bng.get_road_edges('7936')
        # edges.extend(bng.get_road_edges('7836')) #7952
    print("retrieved road edges")
    actual_middle = [edge['middle'] for edge in edges]
    roadmiddle = copy.deepcopy(actual_middle)
    # print(f"{actual_middle[0]=}")
    roadleft = [edge['left'] for edge in edges]
    roadright = [edge['right'] for edge in edges]
    adjusted_middle = [np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle']))/adjustment_factor for edge in edges]
    centerline = actual_middle
    return actual_middle, adjusted_middle, roadleft, roadright

def plot_trajectory(traj, title="Trajectory", label1="AI behavior"):
    global centerline, roadleft, roadright, new_results_dir, default_scenario, default_spawnpoint, qr_positions
    sp = spawn_point(default_scenario, default_spawnpoint)
    x = [t[0] for t in traj]
    y = [t[1] for t in traj]
    plt.plot(x,y, 'b', label=label1)
    # plt.gca().set_aspect('equal')
    # plt.axis('square')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'k-', label="centerline")
    plt.plot([t[0] for t in roadleft], [t[1] for t in roadleft], 'r-', label="left")
    plt.plot([t[0] for t in roadright], [t[1] for t in roadright], 'g-', label="right")
    plt.scatter(sp['pos'][0], sp['pos'][1], marker="o", linewidths=10, label="spawnpoint")
    plt.plot([p[0][0] for p in qr_positions], [p[0][1] for p in qr_positions], linewidth=5, label="billboard")
    plt.title(title)
    plt.legend()
    plt.draw()
    if new_results_dir == '':
        plt.savefig("{}/{}-{}_expected-trajectory.jpg".format(os.getcwd(), default_scenario, default_spawnpoint))
    else:
        plt.savefig("{}/{}-{}_expected-trajectory.jpg".format(new_results_dir, default_scenario, default_spawnpoint))
    #plt.savefig("{}/{}-{}_expected-trajectory.jpg".format(new_results_dir, default_scenario, default_spawnpoint))
    # plt.show()
    # plt.pause(1)
    plt.close("all")


def get_start_index(adjusted_middle):
    global default_scenario, default_spawnpoint
    sp = spawn_point(default_scenario, default_spawnpoint)
    distance_from_centerline = dist_from_line(adjusted_middle, sp['pos'])
    idx = max(np.where(distance_from_centerline == min(distance_from_centerline)))
    return idx[0]


def create_ai_line_from_road_with_interpolation(spawn, bng, swerving=False):
    global centerline, remaining_centerline, centerline_interpolated, roadleft, roadright
    points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
    actual_middle, adjusted_middle, roadleft, roadright = road_analysis(bng)
    print("finished road analysis")
    start_index = get_start_index(adjusted_middle)
    middle_end = adjusted_middle[:start_index]
    middle = adjusted_middle[start_index:]
    # temp = [list(spawn['pos'])]; temp.extend(middle); middle = temp
    middle.extend(middle_end)
    middle.append(middle[0])
    # middle = list(np.roll(np.array(adjusted_middle), 3*(len(adjusted_middle) - 4)))
    # set up centerline with swerving
    print(f"{swerving=}")
    if swerving:
        dists = []
        swerving_middle = []
        swerve_traj = []
        for i,p in enumerate(middle[:-1]):
            dists.append(distance2D(p[:-1], middle[i+1][:-1]))
        for i,p in enumerate(middle[:-1]):
            if dists[i] > 20:
                y_interp = interpolate.interp1d([p[0], middle[i + 1][0]], [p[1], middle[i + 1][1]])
                xs = np.linspace(p[0], middle[i + 1][0], num=10, endpoint=True)
                ys = y_interp(xs)
                swerving_middle.extend([[x,y,p[2]] for x,y in zip(xs,ys)])
            else:
                swerving_middle.append(p)
        swerving_middle.append(middle[0])
        # randomly intersperse swerve points
        middle = [[p[0] + random.random() * 3, p[1] + random.random() * 3, p[2]] for p in swerving_middle if random.random() > 0]
        # temp = []
        # swerve_points = []
        # for p in swerving_middle:
        #     if random.random() > 0:
        #         temp.append(p)
        #         swerve_points.append(False)
        #     else:
        #         temp.append([p[0] + random.random() * 3, p[1] + random.random() * 3, p[2]])
        #         swerve_points.append(True)
        # swerving_middle = temp
        # middle = swerving_middle
        # interpolate centerline
        for i,p in enumerate(middle[:-1]):
            # interpolate at 1m distance
            if distance(p, middle[i+1]) > 1:
                y_interp = interpolate.interp1d([p[0], middle[i + 1][0]], [p[1], middle[i + 1][1]])
                num = int(distance(p, middle[i + 1]))
                xs = np.linspace(p[0], middle[i + 1][0], num=num, endpoint=True)
                ys = y_interp(xs)
                traj.extend([[x,y] for x,y in zip(xs,ys)])
            else:
                traj.append([p[0],p[1]])
        # interpolate swerve line and swerve flags
        for i,p in enumerate(swerving_middle[:-1]):
            # interpolate at 1m distance
            if distance(p, swerving_middle[i+1]) > 1:
                y_interp = interpolate.interp1d([p[0], swerving_middle[i + 1][0]], [p[1], swerving_middle[i + 1][1]])
                num = int(distance(p, swerving_middle[i + 1]))
                xs = np.linspace(p[0], swerving_middle[i + 1][0], num=num, endpoint=True)
                ys = y_interp(xs)
                swerve_traj.extend([[x,y] for x,y in zip(xs,ys)])
            else:
                swerve_traj.append([p[0],p[1]])
        # set up debug line
        for i,p in enumerate(actual_middle[:-1]):
            points.append([p[0], p[1], p[2]])
            point_colors.append([0, 1, 0, 0.1])
            spheres.append([p[0], p[1], p[2], 0.25])
            sphere_colors.append([1, 0, 0, 0.8])
            # count += 1
    else: # not swerving
        for i,p in enumerate(middle[:-1]):
            # interpolate at 1m distance
            if distance(p, middle[i+1]) > 1:
                y_interp = interpolate.interp1d([p[0], middle[i + 1][0]], [p[1], middle[i + 1][1]])
                num = int(distance(p, middle[i + 1]))
                xs = np.linspace(p[0], middle[i + 1][0], num=num, endpoint=True)
                ys = y_interp(xs)
                traj.extend([[x,y,p[2]] for x,y in zip(xs,ys)])
            else:
                traj.append(copy.deepcopy(p))
    print("set up debug line")
    # set up debug line
    for i,p in enumerate(adjusted_middle[:-1]):
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.25])
        sphere_colors.append([1, 0, 0, 0.8])
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    points = []; point_colors=[]; spheres = []; sphere_colors=[]
    for i,p in enumerate(roadleft[:-1]):
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.5])
        sphere_colors.append([0, 1, 0, 0.8])
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    points = []; point_colors=[]; spheres = []; sphere_colors=[]
    for i,p in enumerate(roadright[:-1]):
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 1.0])
        sphere_colors.append([0, 0, 1, 0.8])
    # for i,p in enumerate(roadright[:-1]):
    #     points.append([p[0], p[1], p[2]])
    #     point_colors.append([0, 1, 0, 0.1])
    #     spheres.append([p[0], p[1], p[2], 0.25])
    #     sphere_colors.append([1, 0, 0, 0.8])
    print("spawn point:{}".format(spawn))
    print("beginning of script:{}".format(middle[0]))
    plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
    centerline = copy.deepcopy(traj)
    remaining_centerline = copy.deepcopy(traj)
    centerline_interpolated = copy.deepcopy(traj)
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return bng

# track is approximately 12.50m wide
# car is approximately 1.85m wide
def has_car_left_track(vehicle_pos, vehicle_bbox, bng):
    global centerline_interpolated
    # get nearest road point
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    # check if it went over left edge
    return min(distance_from_centerline) > 9.0 #10,9.5,9.25

def has_car_almost_left_track(vehicle_pos, vehicle_bbox, bng):
    global centerline_interpolated
    # get nearest road point
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    # check if it went over left edge
    mindist = min(distance_from_centerline)
    return mindist > 1.5

def calc_points_of_reachable_set(vehicle_state):
    turn_rad = math.radians(30)
    offset = math.radians(90)
    yaw = vehicle_state['yaw'][0]
    points = []
    radius = 11.1
    # leftmost point
    points.append([vehicle_state['front'][0]+ radius*math.cos(yaw+turn_rad-offset),
                  vehicle_state['front'][1]+ radius*math.sin(yaw+turn_rad-offset),
                  vehicle_state['front'][2]])
    # front point touching car
    points.append(vehicle_state['front'])
    # rightmost point
    points.append([vehicle_state['front'][0] + radius*math.cos(yaw - turn_rad-offset),
                   vehicle_state['front'][1] + radius*math.sin(yaw - turn_rad-offset),
                   vehicle_state['front'][2]])
    return points

def calc_yaw(vehicle_state, vehicle_bbox):
    print(f"{vehicle_state['yaw']=}")

    return

def nearest_seg(road, pos):
    global roadleft, roadright
    road_seg = {}
    dists = dist_from_line(road, pos)
    idx = max(np.where(dists == min(dists)))[0]
    road_seg_left = []
    road_seg_right = []
    # road_seg_center = []
    for i in range(-1,15):
        if idx + i < 0:
            road_seg_left.append(roadleft[len(roadleft) + (idx + i)])
            road_seg_right.append(roadright[len(roadright) + (idx + i)])
        else:
            road_seg_left.append(roadleft[idx+i])
            road_seg_right.append(roadright[idx+i])
    road_seg['left'] = road_seg_left
    road_seg['right'] = road_seg_right
    return road_seg

def intersection_of_RS_and_road(rs, road_seg):
    segpts = copy.deepcopy(road_seg['left'])
    temp = road_seg['right']
    temp.reverse()
    segpts.extend(temp)
    p1 = Polygon([tuple(p[:2]) for p in rs])
    p2 = Polygon([tuple(p[:2]) for p in segpts])
    intersects = p1.intersects(p2)
    if intersects:
        intersect_area = p1.intersection(p2).area
    else:
        intersect_area = 0.0
    rs_area = p1.area
    x = intersect_area / rs_area
    return x

@ignore_warnings
def plot_intersection_with_CV2(vehicle_state, rs, road_seg, intersection, bbox):
    fig, ax = plt.subplots()
    yaw = vehicle_state['yaw'][0]
    patches = []
    radius = 11.1
    # plot LR limits of RS
    plt.plot([p[0] for p in rs], [p[1] for p in rs], "tab:purple", label="reachable set (1 sec.)")
    # plot area of RS
    wedge = Wedge((vehicle_state['front'][0], + vehicle_state['front'][1]), radius, math.degrees(yaw) - 30 - 90, math.degrees(yaw) + 30 - 90)
    patches.append(wedge)
    p = PatchCollection(patches, alpha=0.4)
    colors = np.array([70.0])
    p.set_array(colors)
    ax.add_collection(p)
    x = np.array([bbox[k][0] for k in bbox.keys()])
    y = np.array([bbox[k][1] for k in bbox.keys()])
    plt.plot(x,y, "m", label="car (bounding box)")
    plt.plot([x[0], x[-1]],[y[0], y[-1]], "m", linewidth=1)
    plt.plot([x[4], x[2], x[1], x[-1]], [y[4], y[2], y[1], y[-1]], "m", linewidth=1)
    plt.plot([vehicle_state['front'][0]], [vehicle_state['front'][1]], "ms", label="car (front)")
    # add road segment
    for k in road_seg.keys():
        plt.plot([road_seg[k][i][0] for i in range(len(road_seg[k]))], [road_seg[k][i][1] for i in range(len(road_seg[k]))], "k")
    plt.title(f'Reachable Set Intersection ({intersection*100:.2f}%)')
    plt.legend()
    plt.axis('square')
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')

    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # display image with opencv or any operation you like
    cv2.imshow("plot", img)
    cv2.waitKey(1)
    plt.close('all')

def plot_intersection(vehicle_state, rs, road_seg, intersection, bbox):
    fig, ax = plt.subplots()
    yaw = vehicle_state['yaw'][0]
    patches = []
    radius = 11.1
    # plot LR limits of RS
    plt.plot([p[0] for p in rs], [p[1] for p in rs], "tab:purple", label="reachable set (1 sec.)")
    # plot area of RS
    wedge = Wedge((vehicle_state['front'][0], + vehicle_state['front'][1]), radius, math.degrees(yaw) - 30 - 90, math.degrees(yaw) + 30 - 90)
    patches.append(wedge)
    p = PatchCollection(patches, alpha=0.4)
    colors = np.array([70.0])
    p.set_array(colors)
    ax.add_collection(p)
    x = np.array([bbox[k][0] for k in bbox.keys()])
    y = np.array([bbox[k][1] for k in bbox.keys()])
    plt.plot(x,y, "m", label="car (bounding box)")
    plt.plot([x[0], x[-1]],[y[0], y[-1]], "m", linewidth=1)
    plt.plot([x[4], x[2], x[1], x[-1]], [y[4], y[2], y[1], y[-1]], "m", linewidth=1)
    plt.plot([vehicle_state['front'][0]], [vehicle_state['front'][1]], "ms", label="car (front)")
    # add road segment
    for k in road_seg.keys():
        plt.plot([road_seg[k][i][0] for i in range(len(road_seg[k]))], [road_seg[k][i][1] for i in range(len(road_seg[k]))], "k")
    plt.title(f"Reachable Set Intersection ({intersection}%)")
    plt.legend()
    plt.close("all")


def returned_to_expected_traj(pos_window):
    global expected_trajectory
    dists = []
    for point in pos_window:
        dist = dist_from_line(expected_trajectory, point)
        dists.append(min(dist))
    avg_dist = sum(dists) / len(dists)
    return round(avg_dist, 0) < 1

def find_width_of_road(bng):
    edges = bng.get_road_edges('7983')
    left_edge = [edge['left'] for edge in edges]
    right_edge = [edge['right'] for edge in edges]
    middle = [edge['middle'] for edge in edges]
    dist1 = distance(left_edge[0], middle[0])
    dist2 = distance(right_edge[0], middle[0])
    print("width of road:", (dist1 + dist2))
    return dist1 + dist2

def overlay_transparent_nowarp(img1, img2, x, y):
    arr = np.zeros(img2.shape)
    arr = arr.astype(np.uint8)
    white_bkgrd = Image.fromarray(arr)
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    img2 = img2.convert("RGB")
    img1.paste(white_bkgrd, (x, y))
    img1.paste(img2, (x, y))
    return np.array(img1)

# with warp
def overlay_transparent(img1, img2, corners):
    orig = torch.from_numpy(img1)[None].permute(0, 3, 1, 2) / 255.0
    pert = torch.from_numpy(img2)[None].permute(0, 3, 1, 2) / 255.0

    _, c, h, w = _, *pert_shape = pert.shape
    _, *orig_shape = orig.shape
    patch_coords = corners[None]
    src_coords = np.tile(
        np.array(
            [
                [
                    [0.0, 0.0],
                    [w - 1.0, 0.0],
                    [0.0, h - 1.0],
                    [w - 1.0, h - 1.0],
                ]
            ]
        ),
        (len(patch_coords), 1, 1),
    )
    src_coords = torch.from_numpy(src_coords).float()
    patch_coords = torch.from_numpy(patch_coords).float()

    # build the transforms to and from image patches
    try:
        perspective_transforms = kornia.get_perspective_transform(src_coords, patch_coords)
    except Exception as e:
        print(f"{e=}")
        print(f"{src_coords=}")
        print(f"{patch_coords=}")

    perturbation_warp = kornia.warp_perspective(
        pert,
        perspective_transforms,
        dsize=orig_shape[1:],
        mode="nearest",
        align_corners=True
    )
    mask_patch = torch.ones(1, *pert_shape)
    warp_masks = kornia.warp_perspective(
        mask_patch, perspective_transforms, dsize=orig_shape[1:],
        mode="nearest",
        align_corners=True
    )
    perturbed_img = orig * (1 - warp_masks)
    perturbed_img += perturbation_warp * warp_masks
    return (perturbed_img.permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

def setup_beamng(setup_args, vehicle_model='hopper',
                 model_name="test-7-trad-50epochs-64batch-1e4lr-ORIGDATASET-singleoutput-model-epoch-43.pt",
                 track=default_scenario, spawnpoint=default_spawnpoint):
    global new_results_dir, default_color, steps_per_sec, qr_positions
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0
    model = torch.load(f"{args.path2src}/GitHub/superdeepbillboard/{model_name}", map_location=torch.device('cuda')).eval()
    setup_logging()

    beamng = BeamNGpy('localhost', 64256, home=f'{args.path2src}/BeamNG.research.v1.7.0.1', user=f'{args.path2src}/BeamNG.research')
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle)
    spawn = spawn_point(track, spawnpoint)
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat'])
    add_barriers(scenario)
    add_qr_cubes(scenario)
    # add_qr_cubes(scenario, setup_args['billboard_pos'], setup_args['billboard_quat'])

    # Compile the scenario and place it in BeamNG's map folder
    scenario.make(beamng)

    # Start BeamNG and enter the main loop
    bng = beamng.open(launch=True)
    # bng.hide_hud()
    bng.set_steps_per_second(steps_per_sec)  # With 36hz temporal resolution
    bng.set_deterministic()  # Set simulator to be deterministic

    # bng.set_particles_enabled
    bng.load_scenario(scenario)
    bng.start_scenario()
    # Put simulator in pause awaiting further inputs
    bng.pause()
    assert vehicle.skt
    # find_width_of_road(bng)
    return vehicle, bng, model, spawn

# uses blob detection
def get_qr_corners_from_colorseg_image_nowarp(image):
    image = np.array(image)
    orig_image = copy.deepcopy(image)

    # mask + convert image to inverted greyscale
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (50, 230, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    inverted_img = util.invert(imgGray)
    inverted_img = np.uint8(inverted_img)
    inverted_img = 255 - inverted_img

    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(inverted_img)
    if keypoints == []:
        # print("No QR code detected")
        return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
    else:
        # ORDER: upper left, upper right, lower left, lower right
        bboxes = [[(int(keypoint.pt[0] - keypoint.size / 2), int(keypoint.pt[1] - keypoint.size / 2)),
                   (int(keypoint.pt[0] + keypoint.size / 2), int(keypoint.pt[1] - keypoint.size / 2)),
                   (int(keypoint.pt[0] - keypoint.size / 2), int(keypoint.pt[1] + keypoint.size / 2)),
                   (int(keypoint.pt[0] + keypoint.size / 2), int(keypoint.pt[1] + keypoint.size / 2))] for keypoint in
                  keypoints]
        boxedimg = cv2.rectangle(orig_image, bboxes[0][0], bboxes[0][3], (255, 0, 0), 1)
        cv2.imshow('boxedimg', boxedimg)
        cv2.waitKey(1)
        return bboxes, boxedimg

# uses contour detection
@ignore_warnings
def get_qr_corners_from_colorseg_image(image):
    image = np.array(image)
    cv2.imshow('colorseg', image)
    cv2.waitKey(1)
    # hsv mask image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_color = (50, 230, 0)  # (50, 235, 235) #(0, 200, 0)
    dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    mask = cv2.inRange(hsv_image, light_color, dark_color)
    image = cv2.bitwise_and(image, image, mask=mask)

    # convert image to inverted greyscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    inverted_img = util.invert(imgGray)
    inverted_img = np.uint8(inverted_img)
    inverted_img = 255 - inverted_img
    inverted_img = cv2.GaussianBlur(inverted_img, (3,3), 0) #9

    # contour detection
    ret, thresh = cv2.threshold(inverted_img, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    if contours == [] or np.array(contours).shape[0] < 2:
        return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
    else:
        epsilon = 0.1 * cv2.arcLength(np.float32(contours[1]), True)
        approx = cv2.approxPolyDP(np.float32(contours[1]), epsilon, True)

        contours = np.array([c[0] for c in contours[1]])
        approx = [c[0] for c in approx][:4]
        # contours = contours.reshape((contours.shape[0], 2))
        if len(approx) < 4:
            return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
        def sortClockwise(approx):
            xs = [a[0] for a in approx]
            ys = [a[1] for a in approx]
            center = [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))]
            def sortFxnX(e):
                return e[0]
            def sortFxnY(e):
                return e[1]
            approx = list(approx)
            approx.sort(key=sortFxnX)
            midpt = int(len(approx) / 2)
            leftedge = list(approx[:midpt])
            rightedge = list(approx[midpt:])
            leftedge.sort(key=sortFxnY)
            rightedge.sort(key=sortFxnY)
            approx = [leftedge[0], leftedge[1], rightedge[1], rightedge[0]]
            return approx, leftedge, rightedge, center
        approx, le, re, center = sortClockwise(approx)
        for i,c in enumerate(le):
            cv2.circle(image, tuple([int(x) for x in c]), radius=1, color=(100 + i*20, 0, 0), thickness=2) # blue
        for i,c in enumerate(re):
            cv2.circle(image, tuple([int(x) for x in c]), radius=1, color=(0, 0, 100 + i*20), thickness=2)  # blue
        cv2.circle(image, tuple(center), radius=1, color=(203,192,255), thickness=2)  # lite pink
        if len(approx) > 3:
            cv2.circle(image, tuple([int(x) for x in approx[0]]), radius=1, color=(0, 255, 0), thickness=2) # green
            cv2.circle(image, tuple([int(x) for x in approx[2]]), radius=1, color=(0, 0, 255), thickness=2) # red
            cv2.circle(image, tuple([int(x) for x in approx[3]]), radius=1, color=(255, 255, 255), thickness=2) # white
            cv2.circle(image, tuple([int(x) for x in approx[1]]), radius=1, color=(147,20,255), thickness=2)# pink

        keypoints = [[tuple(approx[0]), tuple(approx[3]),
                      tuple(approx[1]), tuple(approx[2])]]
        return keypoints, image

def is_billboard_fully_viewable(image, qr_corners):
    pixel_epsilon = 20
    # print(f"{qr_corners=}")
    imageheight = image.size[1]
    imagewidth = image.size[0]
    for corners in qr_corners:
        # ORDER: upper left, upper right, lower left, lower right
        # print(f"{corners[0][0]=}\n{corners[0][1]=}\n{corners[3][0]=}\n{corners[3][1]=}")
        # image indices reversed because of course they are because it's fucking opencv (it's WxH ugh)
        if corners[0][0] <= pixel_epsilon or corners[0][1] <= pixel_epsilon or \
                abs(imagewidth - corners[3][0]) <= pixel_epsilon or abs(imageheight - corners[3][1]) <= pixel_epsilon:
            return False
    return True

def add_perturbed_billboard(img, bb, qr_corners):
    # size = (qr_corners[3][0] - qr_corners[0][0], qr_corners[3][1] - qr_corners[0][1])
    # resized_bb = cv2.resize(bb, size)
    # img = overlay_transparent_nowarp(np.array(img), np.array(resized_bb), qr_corners[0][0], qr_corners[0][1])
    img = overlay_transparent(np.array(img), bb, np.asarray(qr_corners))
    return img

def deepbillboard(model, sequence, direction, bb_size=5, iterations=400, noise_level=25, input_divers=False):
    deepbb = DeepBillboard.DeepBillboard(model, sequence, direction)
    img_arr = [hashmap['image'] for hashmap in sequence]
    img_patches = [hashmap['bbox'][0] for hashmap in sequence]
    new_img_patches = []
    for i, patch in enumerate(img_patches):
        temp = [i]
        for tup in patch:
            temp.append(tup[0]);
            temp.append(tup[1])
        new_img_patches.append(copy.deepcopy(temp))
    perturbed_billboard_images = deepbb.perturb_images(img_arr, np.array(new_img_patches), model, bb_size=bb_size, iterations=iterations,
                                                       noise_level=noise_level, input_divers=input_divers)
    return perturbed_billboard_images

def get_percent_of_image(coords, img):
    coords = [tuple(i) for i in coords[0]]
    coords = tuple([coords[0],coords[1],coords[3],coords[2]])
    patch_size = Polygon(coords).area
    img_size = img.size[0] * img.size[1]
    return patch_size / img_size


def superdeepbillboard(model, sequence, direction, steering_vector, bb_size=5, iterations=400, noise_level=25,
                       dist_to_bb=None, last_billboard=None, input_divers=True, loss_fxn='inv23'):
    sdbb = SuperDeepBillboard.SuperDeepBillboard(model, sequence, direction)
    img_arr = [hashmap['image'] for hashmap in sequence]
    img_patches = [hashmap['bbox'][0] for hashmap in sequence]
    new_img_patches = []
    for i, patch in enumerate(img_patches):
        temp = [i]
        for tup in patch:
            temp.append(tup[0]);
            temp.append(tup[1])
        new_img_patches.append(copy.deepcopy(temp))

    if dist_to_bb is not None:
        constraint = 1 #dist_to_bb / 22.0 #math.atan(dist_to_bb / 22.0) # 0.2 # 0.1667
        if direction == "left":
            steering_vector.append(-constraint)  # steering_vector[-1]-0.1
        else:
            steering_vector.append(constraint)  # steering_vector[-1]+0.1
    else:
        constraint = 1.0 # 0.2  # 0.1667
        if direction == "left":
            steering_vector.append(-constraint)  # steering_vector[-1]-0.1
        else:
            steering_vector.append(constraint)  # steering_vector[-1]+0.1
    tensorized_steering_vector = torch.as_tensor(np.array(steering_vector, dtype=np.float64), dtype=torch.float)
    perturbed_billboard_images, y, MAE = sdbb.perturb_images(img_arr, np.array(new_img_patches), model,
                                                        tensorized_steering_vector, bb_size=bb_size,
                                                        iterations=iterations, noise_level=noise_level,
                                                        last_billboard=last_billboard, loss_fxn=loss_fxn, input_divers=input_divers)
    return perturbed_billboard_images, y, MAE

def run_scenario(vehicle, bng, model, spawn, direction, run_number=0,
                                        collect_sequence_results=None,
                                        bb_size=5, iterations=400, noise_level=25, dist_to_bb_cuton=37, resultsdir="images", input_divers=False):
    global centerline, default_spawnpoint, unperturbed_traj, unperturbed_steer
    starttime = time.time()
    sequence, unperturbed_results = run_scenario_to_collect_sequence(vehicle, bng, model, spawn, cuton=dist_to_bb_cuton)
    pert_billboard, ys, MAE_collseq = deepbillboard(model, sequence, direction, bb_size=bb_size, iterations=iterations, noise_level=noise_level, input_divers=input_divers)
    timetorun = time.time() - starttime
    print(f"Time to perturb: {timetorun:.1f}")
    plt.title("dbb final pert_billboard")
    plt.imshow(pert_billboard)
    plt.savefig("{}/pert_billboard-dbb.jpg".format(resultsdir))
    plt.close('all')
    save_image(torch.from_numpy(pert_billboard).permute(2, 0, 1) / 255.0, "{}/dbb_pert_billboard_torchsaveimg.png".format(resultsdir))
    pert_trajs = []
    Ys = []
    keys = ['unperturbed_deviation', 'unperturbed_traj', 'unperturbed_outcome', 'testruns_deviation', 'testruns_trajs', 'testruns_dists', 'testruns_ys',
            'testruns_error', 'testruns_mse', 'testruns_errors', 'testruns_outcomes']
    values = [[] for k in keys]
    results = {key: value for key, value in zip(keys, values)}
    results["time_to_run_technique"] = timetorun
    results['unperturbed_outcome'] = unperturbed_results["outcome"]
    results['unperturbed_dists'] = unperturbed_results['dists']
    results['unperturbed_deviation'] = unperturbed_results['deviation']
    results['unperturbed_traj'] = unperturbed_results['traj']
    results['unperturbed_all_ys'] = unperturbed_results['all_ys']
    results["num_billboards"] = len(sequence)
    results["MAE_collection_sequence"] = MAE_collseq
    for i in range(10):
        runstarttime = time.time()
        perturbed_results = run_scenario_with_perturbed_billboard(vehicle, bng, model, spawn, pert_billboard,
                                                                  run_number=i)
        print(f"Perturbed run {i} took {time.time()-runstarttime:2.2f}sec to finish.")
        results['testruns_deviation'].append(perturbed_results['deviation'])
        results['testruns_dists'].extend(perturbed_results['dists'])
        results['testruns_mse'].append(perturbed_results['mse'])
        results['testruns_error'].append(perturbed_results['error'])
        results['testruns_errors'].extend(perturbed_results['error'])
        results['testruns_outcomes'].append(perturbed_results["outcome"])
        Ys.append(perturbed_results['all_ys'])
        pert_trajs.append(perturbed_results['traj'])
        # results['errors'].extend(perturbed_results['error'])
    results['testruns_trajs'] = pert_trajs
    results['testruns_all_ys'] = Ys
    results['unperturbed_deviation'] = unperturbed_results['deviation']
    results['unperturbed_dists'] = unperturbed_results['dists']
    results['pertrun_all_ys'] = None
    results["unperturbed_all_ys"] = unperturbed_results['all_ys']
    outstring = f"\nRESULTS FOR DBB {model._get_name()} {default_spawnpoint} {direction=} {bb_size=} {iterations=} {noise_level=}: \n" \
                f"Avg. deviation from expected trajectory: \n" \
                f"unperturbed:\t{results['unperturbed_deviation']}\n" \
                f"perturbed:  \t{sum(results['testruns_deviation']) / float(len(results['testruns_deviation']))} \n" \
                f"Avg. distance from expected trajectory:\n" \
                f"unperturbed:\t{sum(results['unperturbed_dists']) / float(len(results['unperturbed_dists']))}\n" \
                f"perturbed:  \t{sum(results['testruns_dists']) / float(len(results['testruns_dists']))}\n" \
                f"Pred. angle error measures:\n" \
                f"mse:      \t{sum(results['testruns_mse']) / float(len(results['testruns_mse']))}\n" \
                f"avg error:\t{sum(results['testruns_errors']) / float(len(results['testruns_errors']))}\n" \
                f"runtime:\t\t{timetorun}\n" \
                f"num_billboards:\t\t{len(sequence)}\n" \
                f"MAE:\t\t{MAE_collseq:.3f}"
    print(outstring)
    return results

def run_scenario_superdeepbillboard(vehicle, bng, model, spawn, direction, dist_to_bb_cuton=26, dist_to_bb_cutoff=0, run_number=0,
                                    collect_sequence_results=None,
                                    bb_size=5, iterations=400, noise_level=25, resultsdir="images", input_divers=True, loss_fxn='inv23'):
    global default_spawnpoint, unperturbed_traj, unperturbed_steer
    starttime = time.time()
    pert_billboards, perturbation_run_results = run_scenario_for_superdeepbillboard(vehicle, bng, model, spawn, direction,
                                                                                bb_size=bb_size, iterations=iterations, noise_level=noise_level,
                                                                                dist_to_bb_cuton=dist_to_bb_cuton, dist_to_bb_cutoff=dist_to_bb_cutoff,
                                                                                input_divers=input_divers, loss_fxn=loss_fxn)
    timetorun = time.time() - starttime
    print("Time to perturb:", timetorun)
    plt.title("sdbb final pert_billboard")
    plt.imshow(pert_billboards[-1])
    plt.savefig("{}/pert_billboard-sdbb-{}-{}-{}.jpg".format(resultsdir, bb_size, iterations, noise_level))
    plt.close('all')
    pert_trajs = []
    Ys = []
    keys = ['testruns_deviation', 'testruns_dists', 'testruns_ys', 'testruns_mse', 'testruns_error', 'testruns_errors', 'testruns_outcomes']
    values = [[] for k in keys]
    results = {key: value for key, value in zip(keys, values)}
    results["time_to_run_technique"] = timetorun
    results['unperturbed_dists'] = collect_sequence_results['dists']
    results['unperturbed_deviation'] = collect_sequence_results['deviation']
    results['unperturbed_traj'] = collect_sequence_results['traj']
    results['unperturbed_all_ys'] = collect_sequence_results['all_ys']
    results['pertrun_all_ys'] = perturbation_run_results['all_ys']
    results['pertrun_outcome'] = perturbation_run_results["outcome"]
    results['pertrun_traj'] = perturbation_run_results['traj']
    results['pertrun_deviation'] = perturbation_run_results['deviation']
    results['pertrun_dist'] = perturbation_run_results['avg_dist']
    results["num_billboards"] = perturbation_run_results["num_billboards"]
    results["MAE_collection_sequence"] = perturbation_run_results["MAE"]
    for i in range(10):
        # print(f"Run number {i}")
        runresults = run_scenario_with_perturbed_billboard(vehicle, bng, model, spawn, pert_billboards[-1], run_number=i,
                                                                                dist_to_bb_cuton=dist_to_bb_cuton, dist_to_bb_cutoff=dist_to_bb_cutoff)
        results['testruns_deviation'].append(runresults['deviation'])
        results['testruns_dists'].extend(runresults['dists'])
        results['testruns_mse'].append(runresults['mse'])
        results['testruns_error'].append(runresults['error'])
        results['testruns_errors'].extend(runresults['error'])
        results['testruns_outcomes'].append(runresults["outcome"])
        Ys.append(runresults['all_ys'])
        pert_trajs.append(runresults['traj'])
    results['testruns_trajs'] = pert_trajs
    results['testruns_all_ys'] = Ys
    outstring = f"\nRESULTS FOR SDBB {model._get_name()} {default_spawnpoint} {direction=} {bb_size=} {iterations=} {noise_level=}: \n" \
                f"Avg. deviation from expected trajectory: \n" \
                f"unperturbed:\t{results['unperturbed_deviation']}\n" \
                f"pert.run: \t{results['pertrun_deviation']}\n" \
                f"test:  \t\t{sum(results['testruns_deviation']) / float(len(results['testruns_deviation']))} \n" \
                f"Avg. distance from expected trajectory:\n" \
                f"unperturbed:\t{sum(results['unperturbed_dists']) / float(len(results['unperturbed_dists']))}\n" \
                f"pert.run:\t\t{results['pertrun_dist']}\n" \
                f"perturbed:  \t{sum(results['testruns_dists']) / float(len(results['testruns_dists']))}\n" \
                f"Pred. angle error measures in test runs:\n" \
                f"mse:      \t\t{sum(results['testruns_mse']) / float(len(results['testruns_mse']))}\n" \
                f"avg error:\t{sum(results['testruns_errors']) / float(len(results['testruns_errors']))}\n" \
                f"testruns outcomes: \t{results['testruns_outcomes']}\n" \
                f"runtime:\t\t{timetorun}\n" \
                f"MAE:\t\t{results['MAE_collection_sequence']:.3f}"
    print(outstring)
    return results


def plot_billboard_ratios(runtimes, percents, detected_runtimes, detected_percents, distances, detected_distances, title):
    global newdir, new_results_dir
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('ratio of billboard to image', color=color)
    lns1 = ax1.plot(runtimes, percents, label="ratio over entire run", color=color)
    lns2 = ax1.plot(detected_runtimes, detected_percents, label="ratio w/ billboard detected", color='tab:orange')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('distance to billboard (M)', color=color)  # we already handled the x-label with ax1
    lns3 = ax2.plot(runtimes, distances, color=color, label="distance over entire run")
    lns4 = ax2.plot(detected_runtimes, detected_distances, color="midnightblue", label="distance w/ billboard detected")
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if "Normal" in title:
        plt.savefig("results/{}/{}-{}-normal_billboard_ratios.jpg".format(newdir, default_scenario, default_spawnpoint))
    else:
        plt.savefig(
            "{}/{}-{}-collection_billboard_ratios.jpg".format(new_results_dir, default_scenario, default_spawnpoint))

    plt.close("all")


def plot_MAEs(distances, percents, detected_distances, detected_percents, unperturbed_predictions, perturbed_predictions, angleerror_distances,
                          title="MAEs"):
    global newdir, new_results_dir
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('distance (M)')
    ax1.set_ylabel('ratio of billboard to image', color=color)
    lns1 = ax1.plot(distances, percents, label="ratio over entire run", color=color)
    lns2 = ax1.plot(detected_distances, detected_percents, label="ratio w/ billboard detected", color='tab:orange')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Steering predictions', color=color)  # we already handled the x-label with ax1
    lns3 = ax2.plot(angleerror_distances, unperturbed_predictions, color=color, label="unpert. predictions")
    # lns4 = ax2.plot(angleerror_distances, perturbed_predictions, color="midnightblue", label="pert. predictions")
    errors = [b-a for a,b in zip(perturbed_predictions, unperturbed_predictions)]
    lns4 = ax2.plot(angleerror_distances, errors, color="blue", label="prediction error")

    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(title)
    ax2.invert_xaxis()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if "Normal" in title:
        plt.savefig("results/{}/{}-{}-normal_billboard_ratios.jpg".format(newdir, default_scenario, default_spawnpoint))
    else:
        plt.savefig(
            "{}/{}-{}-collection_angle_error-wrtdistance.jpg".format(new_results_dir, default_scenario, default_spawnpoint))
    plt.close("all")


def run_scenario_to_collect_sequence(vehicle, bng, model, spawn, cuton=40, device=torch.device('cuda')):
    global new_results_dir, steps_per_sec, expected_trajectory, unperturbed_steer, roadmiddle
    global integral, prev_error, setpoint
    print("run_scenario_to_collect_sequence")
    bng.restart_scenario()
    bng.pause()
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)

    birdseye = sensors['overhead']['colour'].convert('RGB')
    birdseye.save("birdseye.jpg")

    integral, runtime = 0.0, 0.0
    prev_error = setpoint
    damage = sensors['damage']['damage']
    start_time = sensors['timer']['time']
    final_img, outcome = None, None
    kphs, traj = [], []
    ys, all_ys = [], []
    imagecount = 0
    sequence, steering_inputs = [], []
    qrbox_pos = list(qr_positions[0][0])
    percents, detected_percents = [], []
    detected_runtimes, runtimes = [], []
    distances, detected_distances = [], []
    last_steering_from_sim = 0.0
    while damage <= 0:
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_cam']['colour'].convert('RGB')
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        runtime = sensors['timer']['time'] - start_time
        damage = sensors['damage']['damage']
        colorseg_img = sensors['front_cam']['annotation'].convert('RGB')
        qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
        # qr_corners, bbox_img = get_qr_corners_from_colorseg_image_nowarp(colorseg_img)
        dist_to_bb = distance(vehicle.state['pos'],qrbox_pos)
        percent_of_img = get_percent_of_image(qr_corners, image)
        percents.append(percent_of_img)
        runtimes.append(runtime)
        distances.append(dist_to_bb)
        cv2.imshow('car view', np.array(image)[:, :, ::-1])
        cv2.waitKey(1)
        if bbox_img is not None and kph > 29 and dist_to_bb < cuton:
            detected_percents.append(percent_of_img)
            detected_distances.append(dist_to_bb)
            detected_runtimes.append(runtime)
            sequence.append({"image": model.process_image(image)[0], "bbox": qr_corners})
            imagecount += 1
        if kph > 29 and not is_billboard_fully_viewable(image, qr_corners):
            print("Billboard no longer viewable")
            outcome = "R2NT"
            break
        with torch.no_grad():
            prediction = model(model.process_image(image).to(device))

        if bbox_img is not None and kph > 29 and dist_to_bb < cuton:
            ys.append(prediction)

        # control params
        dt = (sensors['timer']['time'] - start_time) - runtime
        steering = float(prediction[0][0])

        steering_inputs.append(steering)
        if abs(steering) > 0.125 and kph > 30:
            setpoint = 30
            throttle = 0
            brake = 0.125
        else:
            setpoint = overall_throttle_setpoint
            brake = 0
        throttle = throttle_PID(kph, dt)

        vehicle.control(throttle=throttle, steering=steering, brake=0.0)

        # collect metrics
        traj.append(vehicle.state['front'])
        kphs.append(kph)
        final_img = image
        all_ys.append(steering)

        rs = calc_points_of_reachable_set(vehicle.state)
        road_seg = nearest_seg(roadmiddle, vehicle.state['front'])
        x = intersection_of_RS_and_road(rs, road_seg)
        # plot_intersection_with_CV2(vehicle.state, rs, road_seg, x, vehicle.get_bbox())

        if damage > 0.0:
            print(f"Damage={damage:.3f}, exiting...")
            outcome = "D={}".format(round(damage,2))
            break
        if has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng):
            print("Left track, exiting...")
            outcome = "LT"
            break

        bng.step(1, wait=True)
        last_steering_from_sim = sensors['electrics']['steering_input']
    unperturbed_steer = steering_inputs
    cv2.destroyAllWindows()
    plot_billboard_ratios(runtimes, percents, detected_runtimes, detected_percents, distances, detected_distances,
                          "Billboard Ratio of Image during Normal Run (No Perturbation)")

    print(f"Sequence collected; {imagecount=}\n")
    print(f"dist_to_bb_cutoff={dist_to_bb}")
    deviation, dists, avg_dist = calc_deviation_from_center(expected_trajectory, traj)
    results = {'runtime': round(runtime, 3), 'damage': damage, 'kphs': kphs, 'traj': traj, 'final_img': final_img,
               'deviation': deviation, 'dists': dists, 'avg_dist': avg_dist, 'ys': ys, "outcome": outcome, "all_ys" : all_ys,
               "dist_to_bb": dist_to_bb
               }
    return sequence, results

def run_scenario_with_perturbed_billboard(vehicle, bng, model, spawn, pert_billboard, run_number=0, device=torch.device('cuda'),
                                                                                dist_to_bb_cuton=None, dist_to_bb_cutoff=None):
    global new_results_dir, expected_trajectory
    global integral, prev_error, setpoint
    model = model.to(device)
    bng.restart_scenario()
    bng.pause()
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)

    integral, runtime = 0.0, 0.0
    prev_error = setpoint;
    kphs, traj = [], []
    damage = sensors['damage']['damage']
    start_time = sensors['timer']['time']
    final_img = None
    perturbed_predictions, unperturbed_predictions = [], []
    sequence, steering_vector = [], []
    pos_window = np.zeros((5, 3))
    billboard_viewable = True;
    outcomestring = ''
    all_ys = []
    runtimes, detected_runtimes=[], []
    percents, detected_percents=[], []
    distances,detected_distances=[], []
    angleerror_distances, angleerror_runtimes = [], []
    qrbox_pos = list(qr_positions[0][0])
    while damage <= 0:
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        damage = sensors['damage']['damage']
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        runtime = sensors['timer']['time'] - start_time
        origimage = sensors['front_cam']['colour'].convert('RGB')
        image = sensors['front_cam']['colour'].convert('RGB')
        colorseg_img = sensors['front_cam']['annotation'].convert('RGB')
        dist_to_bb = distance(vehicle.state['pos'],qrbox_pos)
        qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
        # qr_corners, bbox_img = get_qr_corners_from_colorseg_image_nowarp(colorseg_img)
        percent_of_img = get_percent_of_image(qr_corners, image)
        runtimes.append(runtime); percents.append(percent_of_img)
        distances.append(dist_to_bb)
        if bbox_img is not None and kph > 29:
            detected_runtimes.append(runtime); detected_percents.append(percent_of_img)
            detected_distances.append(dist_to_bb)
            # collect sequence and steering vector so far
            sequence.append({"image": image, "bbox": qr_corners})
            steering_vector.append(unpert_prediction)
            image_pert = add_perturbed_billboard(origimage, pert_billboard, qr_corners[0])
            cv2.imshow('car view', image_pert[:, :, ::-1])
            cv2.waitKey(1)

            billboard_viewable = is_billboard_fully_viewable(origimage, qr_corners)
            # if not is_billboard_fully_viewable(origimage, qr_corners):
            #     print("Billboard no longer viewable")
            #     break

        with torch.no_grad():
            model = model.to(torch.device("cuda"))
            origimg = model.process_image(origimage).to(torch.device("cuda"))
            unpert_prediction = float(model(origimg).cpu()[0][0])
            origimg = origimg.to(torch.device("cpu"))
            if bbox_img is not None and kph > 29:
                deviceimg_pert = model.process_image(image_pert).to(torch.device("cuda"))
                prediction_pert = float(model(deviceimg_pert).cpu()[0][0])
                steering = prediction_pert
            else:
                steering = unpert_prediction
        # control params
        dt = (sensors['timer']['time'] - start_time) - runtime
        all_ys.append(unpert_prediction)
        if abs(unpert_prediction) > 0.2:
            setpoint = 30
        else:
            setpoint = 40
        throttle = throttle_PID(kph, dt)
        # if abs(kph - setpoint) > 10:
        #     vehicle.control(throttle=throttle, steering=0, brake=0.0)
        #     # print(f"steering=0")
        # else:
        #     vehicle.control(throttle=throttle, steering=steering, brake=0.0)
        #     # print(f"{steering=}")
        vehicle.control(throttle=throttle, steering=steering, brake=0.0)

        if bbox_img is not None and kph > 29:
            # origimg = model.process_image(origimage).to(device)
            # unpert_prediction = float(model(origimg).cpu()[0][0])
            unperturbed_predictions.append(unpert_prediction)
            perturbed_predictions.append(prediction_pert)
            angleerror_runtimes.append(runtime)
            angleerror_distances.append(dist_to_bb)
        traj.append(vehicle.state['front'])
        kphs.append(kph)
        final_img = image
        pos_window = np.roll(pos_window, 3)
        pos_window[0] = vehicle.state['pos']
        # stopping conditions
        if damage > 0.0:
            outcomestring = f"D={damage:2.1f}"
            print(f"Damage={damage:.3f} at timestep={runtime:.2f}, exiting...")
            break
        elif has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng):
            outcomestring = f"LT"
            print("Left track, exiting...")
            break
        elif not billboard_viewable and returned_to_expected_traj(pos_window):
            outcomestring = "R2NT"
            print("Returned to normal trajectory, exiting...")
            break
        # elif distance(spawn['pos'], vehicle.state['pos']) > 65 and runtime > 10:
        elif distance(qrbox_pos, vehicle.state['pos']) > 20 and runtime > 10:
            outcomestring = "2FAR"
            print("Too far from sequence, exiting...")
            break
        bng.step(1, wait=True)

    cv2.destroyAllWindows()
    plot_MAEs(distances, percents, detected_distances, detected_percents, unperturbed_predictions,
               perturbed_predictions, angleerror_distances, title=f"Angle Error during Test Run w.r.t. Distance\ncuton={dist_to_bb_cuton}M cutoff={dist_to_bb_cutoff}")
    mse = mean_squared_error(unperturbed_predictions, perturbed_predictions)
    error = np.array(unperturbed_predictions) - np.array(perturbed_predictions)
    deviation, dists, avg_dist = calc_deviation_from_center(expected_trajectory, traj)
    results = {'runtime': round(runtime, 3), 'damage': damage, 'kphs': kphs, 'traj': traj, 'final_img': final_img,
               'deviation': deviation, 'mse': mse, 'dists': dists, 'avg_dist': avg_dist, 'error': error,
               'perturbed_predictions': perturbed_predictions, 'outcome': outcomestring, "all_ys":all_ys
               }
    return results

def distance2D(a,b):
    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))

def law_of_cosines(A, B, C):
    dist_AB = distance2D(A[:2], B[:2])
    dist_BC = distance2D(B[:2], C[:2])
    dist_AC = distance2D(A[:2], C[:2])
    return math.acos((math.pow(dist_AB,2)+ math.pow(dist_AC,2) -math.pow(dist_BC,2)) / (2 * dist_AB * dist_AC))

def car_facing_billboard(vehicle_state):
    global qr_positions
    center_billboard = qr_positions[0][0]
    alpha = law_of_cosines(vehicle_state['front'], vehicle_state['pos'], center_billboard)
    print(f"{math.degrees(alpha)=}")
    return math.degrees(alpha) > 179.0

def run_scenario_for_superdeepbillboard(vehicle, bng, model, spawn, direction, dist_to_bb_cuton=26, dist_to_bb_cutoff=26,
                                        bb_size=5, iterations=100, noise_level=25, input_divers=True, loss_fxn='inv23',
                                        device=torch.device('cuda')):
    global new_results_dir, centerline, expected_trajectory, qr_positions
    global integral, prev_error, setpoint
    integral, runtime = 0.0, 0.0
    prev_error = setpoint
    model = model.to(device)
    bng.restart_scenario()
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)

    kphs, traj = [], []
    damage = sensors['damage']['damage']
    final_img, outcome = None, None
    perturbed_predictions, unperturbed_predictions = [], []
    start_time = sensors['timer']['time']
    sequence, steering_vector, pert_billboards = [], [], []
    ys, all_ys = [], []
    bb_viewed_window = np.ones((10))
    detected_runtimes=[]; detected_percents=[]; runtimes=[];percents=[]; distances = []; detected_distances = []; angleerror_runtimes = []
    qrbox_pos = list(qr_positions[0][0])  # bng.scenario._get_objects_list()[-1]['options']['position'][:2] #
    MAE = 0
    while damage <= 0:
        sensors = bng.poll_sensors(vehicle)
        last_steering_from_sim = sensors["electrics"]["steering_input"]
        dist_to_bb = distance(vehicle.state['pos'],qrbox_pos)
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        damage = sensors['damage']['damage']
        runtime = sensors['timer']['time'] - start_time
        origimage = sensors['front_cam']['colour'].convert('RGB')
        image = sensors['front_cam']['colour'].convert('RGB')
        colorseg_img = sensors['front_cam']['annotation'].convert('RGB')
        print(f"{dist_to_bb=:.2f}")
        qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
        runtimes.append(runtime)
        percent_of_img = get_percent_of_image(qr_corners, image)
        percents.append(percent_of_img)
        distances.append(dist_to_bb)
        # stopping conditions
        if bbox_img is None and kph > 29 and round(dist_to_bb,0) <= dist_to_bb_cuton:
            bb_viewed_window = np.roll(bb_viewed_window,1)
            bb_viewed_window[0] = 0
            if sum(bb_viewed_window) < len(bb_viewed_window) / 2:
                print("Billboard window no longer viewable")
                outcome = "BBNV"
                break
        if bbox_img is not None and kph > 29 and round(dist_to_bb,0) <= dist_to_bb_cuton:
            detected_runtimes.append(runtime)
            detected_distances.append(dist_to_bb)
            detected_percents.append(percent_of_img)
            almost_left_track = has_car_almost_left_track(vehicle.state['front'], vehicle.get_bbox(), bng)
            rs = calc_points_of_reachable_set(vehicle.state)
            road_seg = nearest_seg(roadmiddle, vehicle.state['front'])
            x = intersection_of_RS_and_road(rs, road_seg)
            x = round(x,2)
            print(f"RS overlap={x}")
            plot_intersection_with_CV2(vehicle.state, rs, road_seg, x, vehicle.get_bbox())

            # stopping conditions
            if damage > 0.0:
                print("Damage={} at timestep={}, exiting...".format(damage, round(runtime, 2)))
                outcome = "D={}".format(round(damage, 2))
                break

            # if kph > 29 and has_car_almost_left_track(vehicle.state['front'], vehicle.get_bbox(), bng) and dist_to_bb < dist_to_bb_cuton:
            #     print("Almost left track, exiting...")
            #     outcome = "LT"
            #     break

            if (kph > 29 and abs(x - dist_to_bb_cutoff) <= 0.02) or (kph > 29 and x <= dist_to_bb_cutoff):
                print(f"RS overlap={x}, exiting...")
                outcome = f"RS overlap={x}"
                break

        # Run SuperDeepbillboard
        if bbox_img is not None and kph > 29 and round(dist_to_bb,0) <= dist_to_bb_cuton: # and (steps % 2 < 1):
            sequence.append({"image": model.process_image(image)[0], "bbox": qr_corners, "colorseg_img":model.process_image(colorseg_img)[0]})
            # image.save(f"{new_sampletaking_dir}/sample-{len(sequence)}.jpg")
            # run superdeepbillboard on it to get perturbed billboard
            print(f"{len(sequence)=}, {len(steering_vector)=}")
            pert_billboard, y, MAE = superdeepbillboard(model, sequence, direction, copy.deepcopy(steering_vector),
                                                   bb_size=bb_size, iterations=iterations, noise_level=noise_level,
                                                   dist_to_bb=dist_to_bb, input_divers=input_divers, loss_fxn=loss_fxn)
            steering_vector.append(last_steering_from_sim)
            model = model.to(device)
            ys = y
            pert_billboards.append(copy.deepcopy(pert_billboard))
            image = add_perturbed_billboard(origimage, pert_billboard, qr_corners[0])
            cv2.imshow('car view', image[:, :, ::-1])
            cv2.waitKey(1)
        elif bbox_img is not None and kph > 29 and round(dist_to_bb,0) <= dist_to_bb_cuton:
            image = add_perturbed_billboard(origimage, pert_billboard, qr_corners[0])
        elif bbox_img is not None and kph > 29:
            sequence.append({"image": model.process_image(image)[0], "bbox": qr_corners,
                             "colorseg_img": model.process_image(colorseg_img)[0]})
            steering_vector.append(last_steering_from_sim)

        with torch.no_grad():
            deviceimg = model.process_image(np.asarray(image)).to(device)
            prediction = model(deviceimg)
            prediction = prediction.cpu()

        # control params
        dt = (sensors['timer']['time'] - start_time) - runtime
        steering = float(prediction[0][0])
        if abs(steering) > 0.2:
            setpoint = 30
        else:
            setpoint = 40
        throttle = throttle_PID(kph, dt)
        vehicle.control(throttle=throttle, steering=steering, brake=0.0)
        all_ys.append(steering)

        if bbox_img is not None and kph > 29 and round(dist_to_bb,0) <= dist_to_bb_cuton:
            origimg = model.process_image(np.asarray(origimage)).to(device)
            unperturbed_predictions.append(float(model(origimg).cpu()[0][0]))
            perturbed_predictions.append(steering)
            angleerror_runtimes.append(runtime)
            traj.append(vehicle.state['front'])
            kphs.append(kph)
        final_img = image
        bng.step(1, wait=True)
    plot_billboard_ratios(runtimes, percents, detected_runtimes, detected_percents, distances, detected_distances,
                          title=f"Billboard Ratio of Image during Collection Run\ncuton={dist_to_bb_cuton}M cutoff={dist_to_bb_cutoff}")
    cv2.destroyAllWindows()
    mse = mean_squared_error(unperturbed_predictions, perturbed_predictions)
    deviation, dists, avg_dist = calc_deviation_from_center(centerline, traj)
    results = {'runtime': round(runtime, 3), 'damage': damage, 'kphs': kphs, 'traj': traj, 'final_img': final_img,
               'deviation': deviation, 'mse': mse, 'steering_vector': steering_vector, 'avg_dist': avg_dist, 'ys': ys,
               "outcome":outcome, "all_ys": all_ys, "num_billboards": len(pert_billboards), "MAE": MAE
               }
    print(f"number of billboards: {len(pert_billboards)}")
    return pert_billboards, results

def make_results_dirs(newdir, technique, bbsize, nl, its, cuton, rsc, input_div=None, timestr=None):
    new_results_dir = "results/{}/results-{}-{}-{}-{}-cuton{}-rs{}-inputdiv{}-{}-{}".format(newdir, technique, bbsize, nl, its, cuton, rsc, input_div, timestr, ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)))
    if not os.path.isdir(new_results_dir):
        os.mkdir(new_results_dir)
    training_file = f"{new_results_dir}/results.pickle"
    return new_results_dir, training_file


def main():
    global new_results_dir, newdir, default_scenario, default_spawnpoint, setpoint, integral
    global prev_error, centerline, centerline_interpolated, unperturbed_traj
    global steps_per_sec
    start = time.time()
    setup_args = None #get_sequence_setup(default_spawnpoint)
    direction = "right"
    techniques = ["dbb-orig", "dbb-plus"]
    model_name = "model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    lossname, new_results_dir = '', ''
    bbsizes = [10]
    iterations = [400]
    noiselevels = [15]
    rscs = [0.60]
    cutons = [28]
    input_divs = [False]
    vehicle, bng, model, spawn = setup_beamng(setup_args, vehicle_model='hopper', model_name=model_name)
    steps_per_sec = 15
    samples = 25
    newdir = f"EXPERIMENT1-{default_spawnpoint}-{''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))}"

    if not os.path.isdir("results/{}".format(newdir)):
        os.mkdir("results/{}".format(newdir))
        print(f"Copying script to {os.getcwd()}/{__file__}")
        shutil.copy(f"{os.getcwd()}/{__file__}", f"results/{newdir}")
        shutil.copy(f"{os.getcwd()}/../deepbillboard/SuperDeepBillboard.py", f"results/{newdir}")
    intake_lap_file(f"posefiles/DAVE2v3-lap-trajectory.txt")
    bng = create_ai_line_from_road_with_interpolation(spawn, bng, swerving=False)

    sequence, unperturbed_results = run_scenario_to_collect_sequence(vehicle, bng, model, spawn, 0)
    unperturbed_traj = unperturbed_results['traj']

    for technique in techniques:
        for rsc in rscs:
            for bbsize in bbsizes:
                for its in iterations:
                    for nl in noiselevels:
                        for cuton in cutons:
                            for input_div in input_divs:
                                for i in range(samples):
                                    all_trajs, all_outcomes = [], []
                                    localtime = time.localtime()
                                    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)

                                    if technique == "dbb":
                                        lossname = 'MAE'
                                        # new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, 1000, its, cuton, rsc, timestr)
                                        new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, nl,
                                                                                           its, cuton, rsc, timestr)
                                        results = run_scenario(vehicle, bng, model, spawn, direction,
                                                               collect_sequence_results=unperturbed_results,
                                                               bb_size=bbsize, iterations=its, noise_level=nl,
                                                               dist_to_bb_cuton=cuton,
                                                               resultsdir=new_results_dir)
                                    elif technique == "dbb-orig":
                                        lossname = 'MDirE'
                                        # new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, nl,
                                        #                                                    its, 28, rsc, timestr)
                                        new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, 1000,
                                                                                           its, cuton, rsc, input_div=False,
                                                                                           timestr=timestr)
                                        results = run_scenario(vehicle, bng, model, spawn, direction,
                                                               collect_sequence_results=unperturbed_results,
                                                               bb_size=bbsize, iterations=its, noise_level=1000,
                                                               dist_to_bb_cuton=cuton,
                                                               resultsdir=new_results_dir)
                                    elif technique == "dbb-plus":
                                        lossname = 'MDirE'
                                        new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, nl,
                                                                                           its, cuton, rsc, input_div, timestr)
                                        # new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, 15,
                                        #                                                    its, 40, rsc, timestr)
                                        results = run_scenario(vehicle, bng, model, spawn, direction,
                                                               collect_sequence_results=unperturbed_results,
                                                               bb_size=bbsize, iterations=its, noise_level=nl,
                                                               dist_to_bb_cuton=cuton,
                                                               resultsdir=new_results_dir, input_divers=True)
                                    else:
                                        lossname = 'inv23' # 'MDirE'
                                        new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, nl,
                                                                                           its, cuton, rsc, timestr=timestr, input_div=False)

                                        results = run_scenario_superdeepbillboard(vehicle, bng, model, spawn, direction, collect_sequence_results=unperturbed_results,
                                                                                  bb_size=bbsize, iterations=its, noise_level=nl, dist_to_bb_cuton=cuton,
                                                                                  dist_to_bb_cutoff=rsc, resultsdir=new_results_dir, input_divers=False, loss_fxn=lossname)
                                        all_trajs.append(results["pertrun_traj"])
                                        all_outcomes.append(results["pertrun_outcome"])
                                    # plot_errors(results['testruns_error'], training_file.replace(".pickle", ".png"))
                                    cv2.destroyAllWindows()
                                    all_trajs.extend(results['testruns_trajs'])
                                    all_outcomes.extend(results['testruns_outcomes'])

                                    write_results(training_file, results, all_trajs, unperturbed_traj,
                                                  model._get_name(), technique, direction, lossname, bbsize, its, nl)
                                    # plot_steering(results["unperturbed_all_ys"], results['pertrun_all_ys'],  results["testruns_all_ys"], title="{}-{}-{}-{}-bbsize={}-iters={}-noise denom={}".format(model._get_name(),
                                    #                                                                      technique, direction,
                                    #                                                                      lossname, bbsize, its, nl), resultsdir=new_results_dir)
                                    plot_deviation(all_trajs, unperturbed_traj,
                                                   "{}-{}-{}\nDOF{}-noisevar{}-cuton{}".format(technique, direction,
                                                                                                         lossname, bbsize, nl, cuton),
                                                   centerline_interpolated, roadleft, roadright, all_outcomes, resultsdir=new_results_dir)
                                    plot_deviation(all_trajs, unperturbed_traj,
                                                   "{}-{}-{}\nDOF{}-noisevar{}-cuton{}-ZOOMED".format(technique, direction,
                                                                                               lossname, bbsize, nl, cuton),
                                                   centerline_interpolated, roadleft, roadright, all_outcomes, resultsdir=new_results_dir)
    bng.close()
    # except:
    #     bng.close()
    print(f"Finished in {time.time() - start} seconds")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    warnings.filterwarnings("ignore")
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    main()
