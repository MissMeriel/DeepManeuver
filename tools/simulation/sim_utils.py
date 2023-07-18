from matplotlib import pyplot as plt
import random
import string
import numpy as np
import statistics
import math
import pandas as pd
import torch
import kornia
import cv2
from shapely.geometry import Polygon
from ast import literal_eval
import pickle
from PIL import Image
import skimage
import copy
import os
import warnings
from functools import wraps
from scipy.spatial.transform import Rotation as R
from beamngpy import StaticObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
from beamngpy import ProceduralCube
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection


import numpy as np
from matplotlib import pyplot as plt
import logging, random, string, time, copy, os, sys, shutil
from perturbation_generator import DeepBillboard, DeepManeuver
from beamngpy import BeamNGpy, Scenario, Vehicle, StaticObject, ScenarioObject
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
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

lanewidth = 3.75  # 2.25

def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

def make_results_dirs(newdir, technique, bbsize, nl, its, cuton, rsc, input_div=None, timestr=None):
    new_results_dir = "results/{}/results-{}-{}-{}-{}-cuton{}-rs{}-inputdiv{}-{}-{}".format(newdir, technique, bbsize, nl, its, cuton, rsc, input_div, timestr, ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)))
    if not os.path.isdir(new_results_dir):
        os.mkdir(new_results_dir)
    training_file = f"{new_results_dir}/results.pickle"
    return new_results_dir, training_file

def spawn_point(scenario_locale, spawn_point='default'):
    if scenario_locale == 'industrial':
        # racetrack sequence starting points
        if spawn_point == "curve1":
            # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.0022243109997362,0.0038272379897535,0.92039221525192,-0.39097133278847)}
            return {'pos': (210.314, -44.7132, 42.7758), 'rot': None, 'rot_quat': (0.0020, 0.0050, 0.9202, -0.3914)}
        elif spawn_point == "curve2":
            return {'pos': (319.606, -92.2611, 43.7064), 'rot': None, 'rot_quat': (0.0079, 0.0126, -0.3788, 0.9253)}
        #     return {'pos':(323.432,-92.7588,43.6475), 'rot': None, 'rot_quat':(0.008327,0.0137599,-0.36539,0.930714)}
        #     return {'pos': (331.169, -104.166, 44.142), 'rot': None, 'rot_quat': (0.009578, 0.033658, -0.359433, 0.93251)}
        elif spawn_point == "curve3":
            return {'pos': (270.271, -206.988, 43.9982), 'rot': None, 'rot_quat': (-0.0072, 0.0055, 0.4767, 0.8790)}
        elif spawn_point == "scenario1" or spawn_point == "scenario2":
            # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.0022243109997362,0.0038272379897535,0.92039221525192,-0.39097133278847)}
            # return {'pos': (252.028,-24.7376,42.814), 'rot': None,'rot_quat': (-0.044106796383858,0.05715386942029,-0.49562504887581,0.8655309677124)} # 130 steps
            return {'pos': (257.414, -27.9716, 42.8266), 'rot': None, 'rot_quat': (-0.0324, 0.0535, -0.4510, 0.8903)} # 50 steps
            # return {'pos': (265.087, -33.7904, 42.805), 'rot': None, 'rot_quat': (-0.022659547626972, 0.023112617433071, -0.42281490564346, 0.9056)} # 4 steps
        elif spawn_point == "scenario4" or spawn_point == "curve4":  # formerly topo21 & topo2, beginning of big curve right before starting gate
            # return {'pos': (171.522, -109.746, 44.8168), 'rot': None, 'rot_quat': (0.0202, -0.0233, 0.1814, 0.9829)} # 50kph
            # return {'pos': (169.177, -115.208, 44.4244), 'rot': None, 'rot_quat': (0.0230, -0.0198, 0.1960, 0.9801)}
            return {'pos': (166.3, -124.9, 43.9), 'rot': None, 'rot_quat': (-0.0237, -0.0283, 0.1419, 0.9892)} # 40kph
        elif spawn_point == "scenario3" :
            return {'pos': (170.179,-111.787,44.6669), 'rot': None, 'rot_quat': (-0.0237, -0.0283, 0.1419, 0.9892)}
        elif spawn_point == "scenario5" or spawn_point == "scenario6":
            return {"pos": (213.948, -192.782, 44.9482), "rot": None, "rot_quat": (0.0057, 0.0227, 0.9882, 0.1512)}
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


def setup_sensors(vehicle, pos=(-0.5, 0.38, 1.3), direction=(0, 1.0, 0)):
    fov = 50
    resolution = (240, 135)
    front_camera = Camera(pos, direction, fov, resolution, colour=True, depth=True, annotation=True)
    # gforces = GForces()
    # electrics = Electrics()
    # damage = Damage()
    # timer = Timer()
    vehicle.attach_sensor('front_cam', front_camera)
    vehicle.attach_sensor('gforces', GForces())
    vehicle.attach_sensor('electrics', Electrics())
    vehicle.attach_sensor('damage', Damage())
    vehicle.attach_sensor('timer', Timer())
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
    return scenario


def add_qr_cubes(scenario, default_scenario, default_spawnpoint):
    qr_positions = []
    posefile = f'posefiles/qr_box_locations-{default_spawnpoint.lower()}.txt'
    with open(posefile, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
            # box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
            #                      rot_quat=rot_quat, scale=(10, 1, 5), JBeam='qrbox2', datablock="default_vehicle")
            # scale=(width, depth, height)
            # box = StaticObject(name='qrbox_{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(3, 0.1, 3),
            #                     shape='vehicles/metal_box/metal_box.dae')
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
        elif default_scenario == "industrial" and default_spawnpoint == "scenario1":
            cube = ProceduralCube(name='cube_platform',
                                  # pos=(145.214,-160.72,43.7269),
                                  pos=(150, -170, 44),
                                  rot=None,
                                  rot_quat=(0, 0, 0, 1),
                                  size=(2, 6, 0.5))
            scenario.add_procedural_mesh(cube)
        elif default_scenario == "industrial" and default_spawnpoint == "scenario2":
            cube = ProceduralCube(name='cube_platform',
                                  pos=(301.259,-83.0026,42.8405),
                                  rot=None,
                                  rot_quat=(0.0016627161530778,0.00650954246521,-0.35758674144745,0.93385589122772),
                                  size=(2, 5, 0.5))
            scenario.add_procedural_mesh(cube)
        elif default_scenario == "industrial" and default_spawnpoint == "scenario3":
            cube = ProceduralCube(name='cube_platform',
                                  pos=(162.416, -163.437, 44.0351),
                                  rot=None,
                                  rot_quat=(0.016658393666148, 0.0089397598057985, 0.02481073141098, 0.99951338768005),
                                  size=(2, 5, 0.5))
            scenario.add_procedural_mesh(cube)
        elif default_scenario == "industrial" and default_spawnpoint == "scenario4":
            cube = ProceduralCube(name='cube_platform',
                                  # pos=(145.214,-160.72,43.7269),
                                  pos=(150, -170, 44),
                                  rot=None,
                                  rot_quat=(0, 0, 0, 1),
                                  size=(2, 6, 0.5))
            scenario.add_procedural_mesh(cube)
        elif default_scenario == "industrial" and default_spawnpoint == "scenario5":
            cube = ProceduralCube(name='cube_platform',
                                  # pos=(183.7,-110.486,44.2),
                                  pos = (185.7,-130.0,44.2),
                                  rot=None,
                                  rot_quat=(0.0,0.0,0.9943694943814233,0.10596843229770164),
                                  size=(2.5, 5, 1.5))
            scenario.add_procedural_mesh(cube)
        elif default_scenario == "industrial" and default_spawnpoint == "scenario6":
            cube = ProceduralCube(name='cube_platform',
                                  pos=(209.449,-134.32,43.3),
                                  rot=None,
                                  rot_quat=(0.0035336871516934865, -0.039940145414387555, -0.9781412912528431, -0.20403899672761666),
                                  size=(2.5, 5, 1.5))
            scenario.add_procedural_mesh(cube)
        elif default_scenario == "driver_training" and default_spawnpoint == "approachingfork":
            cube = ProceduralCube(name='cube_platform',
                                  pos=(-20.3113, 218.448, 50.043),
                                  rot=None,
                                  rot_quat=(-0.022064134478569,-0.022462423890829,0.82797580957413,0.55987912416458),
                                  size=(4, 8, 0.5))
            scenario.add_procedural_mesh(cube)
    return qr_positions


def get_outcomes(results):
    outcomes_counts = {"D":0, "LT":0, "R2NT":0, "2FAR":0}
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


def intake_lap_file(filename="posefiles/DAVE2v3-lap-trajectory.txt"):
    expected_trajectory = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = literal_eval(line)
            expected_trajectory.append(line)
    return expected_trajectory

def get_sequence_setup(sequence):
    df = pd.read_csv("posefiles/sequence-setup.txt", sep="\s+")
    keys = df.keys()
    index = df.index[df['sequence'] == sequence].tolist()[0]
    vals = {key: df.at[index, key] for key in keys}
    print(vals)
    return vals


def get_start_index(adjusted_middle, default_scenario, default_spawnpoint):
    sp = spawn_point(default_scenario, default_spawnpoint)
    distance_from_centerline = dist_from_line(adjusted_middle, sp['pos'])
    idx = max(np.where(distance_from_centerline == min(distance_from_centerline)))
    return idx[0]


# track is approximately 12.50m wide
# car is approximately 1.85m wide
def has_car_left_track(vehicle_pos, centerline_interpolated):
    # get nearest road point
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    # check if it went over left edge
    return min(distance_from_centerline) > 9.0 #10,9.5,9.25

def has_car_almost_left_track(vehicle_pos, centerline_interpolated):
    # get nearest road point
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    # check if it went over left edge
    mindist = min(distance_from_centerline)
    return mindist > 1.5

def nearest_seg(road, pos, roadleft, roadright):
    road_seg = {}
    dists = dist_from_line(road, pos)
    idx = max(np.where(dists == min(dists)))[0]
    road_seg_left = []
    road_seg_right = []
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

def plot_racetrack_roads(roads, bng, default_scenario, default_spawnpoint):
    # global default_scenario, default_spawnpoint
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    print(f"{len(roads)=}")
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp = []
        y_temp = []
        xy_def = [edge['middle'][:2] for edge in road_edges]
        dists = [distance2D(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
        s = sum(dists)
        if (s < 100):
            continue
        for edge in road_edges:
            x_temp.append(edge['middle'][0])
            y_temp.append(edge['middle'][1])
        symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
        plt.plot(x_temp, y_temp, symb, label=road)
    plt.legend(fontsize=8)
    plt.title("{} {}".format(default_scenario, default_spawnpoint))
    plt.show()
    plt.pause(0.001)


def plot_MAEs(distances, percents, detected_distances, detected_percents, unperturbed_predictions, perturbed_predictions, angleerror_distances,
                          title="MAEs", new_results_dir="./new_results", default_scenario="industrial", default_spawnpoint="spawn"):
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
    plt.savefig(
        "{}/{}-{}-collection_angle_error-wrtdistance.jpg".format(new_results_dir, default_scenario, default_spawnpoint))
    plt.close("all")



def plot_billboard_ratios(runtimes, percents, detected_runtimes, detected_percents, distances, detected_distances, title,
                          newdir, new_results_dir, default_scenario="industrial", default_spawnpoint="spawn"):
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


def plot_deviation(trajectories, unperturbed_traj, qr_positions, model, centerline, left, right, outcomes,
                   xlim=[100, 350], ylim=[-260, 0], resultsdir="images", default_scenario="industrial", default_spawnpoint="spawn"):
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
    if default_spawnpoint == "scenario1" and "ZOOMED" not in model:
        plt.xlim([245, 335])
        plt.ylim([-123, -20])
    elif default_spawnpoint == "scenario1" and "ZOOMED" in model:
        plt.xlim([265, 300])
        plt.ylim([-60, -30])
    elif default_spawnpoint == "curve1":
        plt.xlim([265, 300])
        plt.ylim([-60, -30])
    elif default_spawnpoint == "curve2":
        plt.xlim([325, 345])
        plt.ylim([-120, -100])
    elif default_spawnpoint == "curve4" or default_spawnpoint == "topo21" :
        plt.xlim([180, 120])
        plt.ylim([-120, -180])
    elif default_scenario == "driver_training" and default_spawnpoint == "approachingfork":
        plt.xlim([-50, 55])
        plt.ylim([150, 255])
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



def plot_trajectory(traj, centerline, roadleft, roadright, default_scenario, default_spawnpoint, qr_positions,
                    new_results_dir='', title="Trajectory", label1="AI behavior"):
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
    inverted_img = skimage.util.invert(imgGray)
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
    inverted_img = skimage.util.invert(imgGray)
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

def get_percent_of_image(coords, img):
    coords = [tuple(i) for i in coords[0]]
    coords = tuple([coords[0],coords[1],coords[3],coords[2]])
    patch_size = Polygon(coords).area
    img_size = img.size[0] * img.size[1]
    return patch_size / img_size

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


def distance2D(a,b):
    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))

def law_of_cosines(A, B, C):
    dist_AB = distance2D(A[:2], B[:2])
    dist_BC = distance2D(B[:2], C[:2])
    dist_AC = distance2D(A[:2], C[:2])
    return math.acos((math.pow(dist_AB,2)+ math.pow(dist_AC,2) -math.pow(dist_BC,2)) / (2 * dist_AB * dist_AC))


def car_facing_billboard(vehicle_state, qr_positions):
    center_billboard = qr_positions[0][0]
    alpha = law_of_cosines(vehicle_state['front'], vehicle_state['pos'], center_billboard)
    print(f"{math.degrees(alpha)=}")
    return math.degrees(alpha) > 179.0


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


def returned_to_expected_traj(pos_window, expected_trajectory):
    dists = []
    for point in pos_window:
        dist = dist_from_line(expected_trajectory, point)
        dists.append(min(dist))
    avg_dist = sum(dists) / len(dists)
    return round(avg_dist, 0) < 1

def plot_deviation(trajectories, unperturbed_traj, model, centerline, left, right, outcomes, qr_positions,
                   default_scenario, default_spawnpoint,
                   xlim=[100, 350], ylim=[-260, 0], resultsdir="images"):
    # global qr_positions
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
        approx = [c[0] for c in approx]
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

def get_percent_of_image(coords, img):
    coords = [tuple(i) for i in coords[0]]
    coords = tuple([coords[0],coords[1],coords[3],coords[2]])
    patch_size = Polygon(coords).area
    img_size = img.size[0] * img.size[1]
    return patch_size / img_size


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


def find_width_of_road(bng):
    edges = bng.get_road_edges('7983')
    left_edge = [edge['left'] for edge in edges]
    right_edge = [edge['right'] for edge in edges]
    middle = [edge['middle'] for edge in edges]
    dist1 = distance(left_edge[0], middle[0])
    dist2 = distance(right_edge[0], middle[0])
    print("width of road:", (dist1 + dist2))
    return dist1 + dist2

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



def plot_racetrack_roads(roads, bng, default_scenario, default_spawnpoint):
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    print(f"{len(roads)=}")
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp = []
        y_temp = []
        xy_def = [edge['middle'][:2] for edge in road_edges]
        dists = [distance2D(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
        s = sum(dists)
        if (s < 100):
            continue
        for edge in road_edges:
            x_temp.append(edge['middle'][0])
            y_temp.append(edge['middle'][1])
        symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
        plt.plot(x_temp, y_temp, symb, label=road)
    plt.legend(fontsize=8)
    plt.title("{} {}".format(default_scenario, default_spawnpoint))
    plt.show()
    plt.pause(0.001)