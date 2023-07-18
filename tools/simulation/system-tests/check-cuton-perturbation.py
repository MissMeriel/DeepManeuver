import sys

# sys.path.append('H:/GitHub/DAVE2-Keras')
# sys.path.append('H:/GitHub/superdeepbillboard')
# sys.path.append('H:/GitHub/BeamNGpy-meriels-ext')
# sys.path.append('H:/GitHub/BeamNGpy-meriels-ext/src')

import numpy as np
from matplotlib import pyplot as plt
import logging, random, string, time, copy, os

from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy import ProceduralCylinder, ProceduralCone, ProceduralCube, ProceduralBump, ProceduralRing
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

import statistics, math
from scipy.spatial.transform import Rotation as R
from ast import literal_eval
from scipy import interpolate

from superdeepbillboard.deepbillboard import DeepBillboard, SuperDeepBillboard, GradientAscent

import torch
import cv2
from skimage import util
from PIL import Image
from sklearn.metrics import mean_squared_error
import kornia
from torchvision.utils import save_image
import pandas as pd
import pickle

# globals
default_color = 'White'  # 'Red'
default_scenario = 'industrial'  # 'automation_test_track'
default_spawnpoint = "straight1"  # "curve1" #
integral = 0.0
prev_error = 0.0
setpoint = 40  # 50.0 #53.3 #https://en.wikipedia.org/wiki/Speed_limits_by_country
lanewidth = 3.75  # 2.25
centerline = []
centerline_interpolated = []
expected_trajectory = []
steps_per_sec = 30  # prev: 60, 100
base_filename = 'H:/BeamNG_DeepBillboard/sequences/'
qr_positions = []
unperturbed_traj = []
unperturbed_steer = []
new_results_dir = ""

def spawn_point(scenario_locale, spawn_point='default'):
    global lanewidth
    # elif scenario_locale == 'industrial':
    # racetrack sequence starting points
    if spawn_point == "curve1":
        # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.0022243109997362,0.0038272379897535,0.92039221525192,-0.39097133278847)}
        return {'pos': (210.314, -44.7132, 42.7758), 'rot': None,
                'rot_quat': (0.0020199827849865, 0.0049774856306612, 0.92020887136459, -0.3913908302784)}
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
    elif spawn_point == "curve2":
        # return {'pos':(323.432,-92.7588,43.6475), 'rot': None, 'rot_quat':(0.0083266003057361,0.013759891502559,-0.36539402604103,0.93071401119232)}
        # 172.713|E|libbeamng.lua.V.updateGFX|Object position: vec3(327.801,-100.41,43.9318)
        # 172.716|E|libbeamng.lua.V.updateGFX|Object rotation (quat): quat(0.0087151182815433,0.020582119002938,-0.36003017425537,0.93267297744751)
        return {'pos': (331.169, -104.166, 44.142), 'rot': None,
                'rot_quat': (0.0095777017995715, 0.033657912164927, -0.35943350195885, 0.93251436948776)}
    # elif spawn_point == "straight2":
    # elif spawn_point == "curve3":
    # elif spawn_point == "straight3":
    # elif spawn_point == "curve4":
    # elif spawn_point == "straight4":


def get_sequence_setup(sequence):
    df = pd.read_csv("sequence-setup.txt", sep="\s+")
    keys = df.keys()
    index = df.index[df['sequence'] == sequence].tolist()[0]
    vals = {key: df.at[index, key] for key in keys}
    print(vals)
    return vals


def setup_sensors(vehicle, pos=(-0.5, 0.38, 1.3), dir=(0, 1.0, 0)):
    fov = 50  # 60 works for full lap #63 breaks on hairpin turn
    resolution = (240, 135)  # (200, 150) (320, 180) #(1280,960) #(512, 512)
    front_camera = Camera(pos, dir, fov, resolution,
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


def add_barriers(scenario):
    with open('industrial_racetrack_barrier_locations.txt', 'r') as f:
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
    with open('qr_box_locations.txt', 'r') as f:
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
            # cube = ProceduralCube(name='cube',
            #                       pos=pos,
            #                       rot=None,
            #                       rot_quat=rot_quat,
            #                       size=(5, 2, 3))
            # cube.type = 'BeamNGVehicle'
            # scenario.add_procedural_mesh(cube)
            scenario.add_object(box)


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


def plot_deviation(trajectories, unperturbed_traj, model, centerline, left, right, outcomes, xlim=[245, 335], ylim=[-123, -20], resultsdir="images"):
    global qr_positions
    x = [point[0] for point in unperturbed_traj]
    y = [point[1] for point in unperturbed_traj]
    # for point in unperturbed_traj:
    #     x.append(point[0])
    #     y.append(point[1])
    plt.plot(x, y, label="Unperturbed", linewidth=10)
    x = []
    y = []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    for point in left:
        x.append(point[0])
        y.append(point[1])
    for point in right:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, label="road")
    for i, t in enumerate(zip(trajectories, outcomes)):
        x = [point[0] for point in t[0]]
        y = [point[1] for point in t[0]]
        if i == 0 and 'sdbb' in model:
            plt.plot(x, y, label="Pert. Run {} ({})".format(i, t[1]), linewidth=5)
        else:
            plt.plot(x, y, label="Pert. Run {} ({})".format(i, t[1]), alpha=0.75)
        i += 1
    for position in qr_positions:
        plt.plot(position[0][0], position[0][1], 'ro', linewidth=5)
    plt.title('Trajectories with {}'.format(model))
    # plt.legend()
    # if default_spawnpoint == "straight1":
    #     plt.xlim([245, 335])
    #     plt.ylim([-123, -20])
    # elif default_spawnpoint == "curve2":
    #     plt.xlim([325, 345])
    #     plt.ylim([-120, -100])
    plt.xlim(xlim)
    plt.ylim(ylim)
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    plt.savefig("{}/{}-{}.jpg".format(resultsdir, model.replace("\n", "-"), randstr))
    plt.show()
    plt.pause(0.1)


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
    plt.show()
    plt.pause(0.1)


def plot_errors(errors, filename="images/errors.png"):
    plt.title("Errors")
    for ei, e in enumerate(errors):
        plt.plot(range(len(e)), e, label=f"Error {ei}")
    plt.savefig("{}".format(filename))
    plt.show()
    plt.pause(0.1)
    plt.title("Error Distributions per Run")
    avgs = []
    for ei, e in enumerate(errors):
        plt.scatter(np.ones((len(e)))*ei, e, s=5)
        avgs.append(float(sum(e)) / len(e))
    plt.plot(range(len(avgs)), avgs)
    plt.savefig("{}".format(filename.replace(".png", "-distribution.png")))
    plt.show()
    plt.pause(0.1)

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


def intake_lap_file(filename="DAVE2v1-lap-trajectory.txt"):
    global expected_trajectory
    expected_trajectory = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = literal_eval(line)
            expected_trajectory.append(line)


def road_analysis(bng):
    # roads = bng.get_roads()
    # get relevant road
    edges = bng.get_road_edges('7983')
    middle = [edge['middle'] for edge in edges]
    left = [edge['left'] for edge in edges]
    right = [edge['right'] for edge in edges]
    return middle, left, right


def create_ai_line_from_road_with_interpolation():
    global centerline, centerline_interpolated
    line = []
    points = []
    traj = []
    middle = copy.deepcopy(centerline)
    middle_end = middle[:3]
    middle = middle[3:]
    middle.extend(middle_end)
    timestep = 0.1
    elapsed_time = 0
    count = 0
    for i, p in enumerate(middle[:-1]):
        # interpolate at 1m distance
        if distance(p, middle[i + 1]) > 1:
            y_interp = interpolate.interp1d([p[0], middle[i + 1][0]], [p[1], middle[i + 1][1]])
            num = int(distance(p, middle[i + 1]))
            xs = np.linspace(p[0], middle[i + 1][0], num=num, endpoint=True)
            ys = y_interp(xs)
            for x, y in zip(xs, ys):
                traj.append([x, y])
                line.append({"x": x, "y": y, "z": p[2], "t": i * timestep})
                points.append([x, y, p[2]])
        else:
            elapsed_time += distance(p, middle[i + 1]) / 12
            traj.append([p[0], p[1]])
            line.append({"x": p[0], "y": p[1], "z": p[2], "t": elapsed_time})
            points.append([p[0], p[1], p[2]])
            count += 1
    centerline_interpolated = points


# track is approximately 12.50m wide
# car is approximately 1.85m wide
def has_car_left_track(vehicle_pos, vehicle_bbox, bng):
    global centerline_interpolated
    # get nearest road point
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    # check if it went over left edge
    return min(distance_from_centerline) > 10


def returned_to_expected_traj(pos_window):
    global expected_trajectory
    dists = []
    for point in pos_window:
        dist = dist_from_line(expected_trajectory, point)
        dists.append(min(dist))
    avg_dist = sum(dists) / len(dists)
    if avg_dist < 1:
        return True
    return False


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
    # img2 = img2.convert("RGBA")
    # img1.paste(img2, (x,y), mask=img2)
    img2 = img2.convert("RGB")
    img1.paste(white_bkgrd, (x, y))
    img1.paste(img2, (x, y))
    # img1.show()
    # print(f"{img1=}\n{np.array(img1).shape=}")
    return np.array(img1)


# with warp
def overlay_transparent(img1, img2, corners):
    # print(img1.min(), img1.max(), img2.min(), img2.max())
    # print(img1.shape, img2.shape)
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
    global base_filename, default_color, steps_per_sec, qr_positions
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0
    # model = torch.load(f"H:/GitHub/DAVE2-Keras/{model_name}", map_location=torch.device('cpu')).eval()
    model = torch.load(f"H:/GitHub/DAVE2-Keras/{model_name}", map_location=torch.device('cuda')).eval()

    # random.seed(1703)
    setup_logging()
    # sequence
    # direction
    # spawn_pos
    # spawn_quat
    # billboard_pos
    # billboard_quat

    beamng = BeamNGpy('localhost', 64256, home='H:/BeamNG.research.v1.7.0.1clean', user='H:/BeamNG.research')
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle)
    spawn = spawn_point(track, spawnpoint)
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat'])
    # scenario.add_vehicle(vehicle, pos=setup_args['spawn_pos'], rot=None, rot_quat=setup_args['spawn_quat'])
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
    # image.save("colorsegimg.jpg")
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
def get_qr_corners_from_colorseg_image(image):
    # image.save("colorsegimg.jpg")
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
    # kernel = np.ones((5, 5), np.float32) / 30
    # inverted_img = cv2.filter2D(inverted_img, -1, kernel)
    # # plt.title("inverted_img")
    # # plt.imshow(inverted_img, cmap="gray")
    # # plt.show()
    # # plt.pause(0.01)
    #
    # contour detection
    ret, thresh = cv2.threshold(inverted_img, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # print(f"{contours=}")
    # print(f"{len(contours[1])=}")
    if contours == [] or np.array(contours).shape[0] < 2:
        return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
    else:
        epsilon = 0.1 * cv2.arcLength(np.float32(contours[1]), True)
        approx = cv2.approxPolyDP(np.float32(contours[1]), epsilon, True)
        # print(f"aPPROXIMATED POLYGON VERTICES {len(approx)=}")
        # hull = []
        # # print(f"{len(contours[1])=}")
        # for i in range(len(contours[1])):
        #     hull.append(cv2.convexHull(contours[1][i], False))
        # print(hull)
        # approx = cv2.convexHull(np.float32(contours[1]), )
        # draw contours on the original image
        # image_copy = inverted_img.copy()
        # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
        #                  lineType=cv2.LINE_AA)
        # cv2.imshow('Contoured image', image_copy)
        # cv2.waitKey(1)
        # plt.title("contours")
        # plt.imshow(image_copy, cmap="gray")
        # plt.show()
        # plt.pause(0.01)

        # ORDER: upper left, upper right, lower left, lower right
        # print(f"{contours=}")
        # contours = np.array(contours[1])
        # contours = contours.reshape((contours.shape[0], 2))

        contours = np.array([c[0] for c in contours[1]])
        approx = [c[0] for c in approx]
        contours = contours.reshape((contours.shape[0], 2))
        # percentage = round(((bboxes[0][3][0]-bboxes[0][0][0])*(bboxes[0][3][1]-bboxes[0][0][1]))/(150*200)*100,3)
        # print(f"{len(approx)=}")
        # print(f"{approx=}")
        # print(f"{type(approx)=}")
        # def sortAll(a):
        #     smallest_x = a.index(min([i[0][0] for i in a]))
        #     return a
        if len(approx) < 4:
            return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
        # print(f"{approx=}")
        def sortClockwise(approx):
            xs = [a[0] for a in approx]
            ys = [a[1] for a in approx]
            center = [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))]
            def sortFxnX(e):
                return e[0]
            def sortFxnY(e):
                return e[1]
            # def sortDist(e):
            #     return math.sqrt(math.pow(e[0]-center[0],2)+math.pow(e[1]-center[1],2))
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

        # plt.title("points")
        # plt.imshow(image)
        # plt.show()
        # plt.pause(0.01)
        # cv2.imshow('Corner points', image)
        # cv2.waitKey(1)
        # plt.title("bboxed img \n(occupies {}% of pixels)".format(percentage))
        # plt.imshow(boxedimg)
        # plt.pause(0.01)
        # print("point sizes:", [keypoint.size for keypoint in keypoints], "\n")
        # keypoints = [[tuple(approx[0]), tuple(approx[1]),
        #               tuple(approx[3]), tuple(approx[2])]]
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


def gradientascent(model, sequence, direction, bb_size=5, iterations=400, noise_level=25, outdir=None):
    deepbb = GradientAscent.GradientAscent(model, sequence, direction, outdir)
    img_arr = [hashmap['image'] for hashmap in sequence]
    img_patches = [hashmap['bbox'][0] for hashmap in sequence]
    new_img_patches = []
    for i, patch in enumerate(img_patches):
        temp = [i]
        for tup in patch:
            temp.append(tup[0]);
            temp.append(tup[1])
        new_img_patches.append(copy.deepcopy(temp))
    perturbed_billboard_images, ys, perturbed_images, all_perturbed_images = deepbb.perturb_images(img_arr, np.array(new_img_patches), model, bb_size=bb_size, iterations=iterations, noise_level=noise_level)
    return perturbed_billboard_images, ys, perturbed_images, all_perturbed_images


def get_percent_of_image(bbox, img):
    try:
        patch_size = np.array(bbox[0][3]) - np.array(bbox[0][0])
        patch_size = float(patch_size[0] * patch_size[1])
        img_size = img.shape[0] * img.shape[1]
        return patch_size / img_size, patch_size
    except:
        return 0,0


def run_scenario(vehicle, bng, model, spawn, direction, run_number=0,
                                        collect_sequence_results=None,
                                        bb_size=5, iterations=400, noise_level=25, resultsdir="images"):
    global centerline, default_spawnpoint, unperturbed_traj, unperturbed_steer
    starttime = time.time()
    sequence, unperturbed_results = run_scenario_to_collect_sequence(vehicle, bng, model, spawn, direction=direction, run_number=run_number, outdir=resultsdir,
                                                                     bb_size=bb_size, iterations=iterations, noise_level=noise_level)
    timetorun = time.time() - starttime
    print("Time to perturb:", timetorun)

    with open(f"{new_results_dir}/inputs.txt", 'w') as f:
        f.write("DISTANCE,ORIG_INPUT,PERT_INPUT,ERROR\n")
        for d,orig,pert,err in zip(unperturbed_results['distances'], unperturbed_results['inputs_orig'],unperturbed_results['inputs_pert'],unperturbed_results['errors']):
            f.write(f"{d:.2f},\t{orig[0][0]:.2f},\t{pert:.2f},\t{err[0][0]:.3f}\n")
    return unperturbed_results

def plot_errors_by_cuton(distances, errors, title):
        global newdir, new_results_dir
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('distance to billboard (M)')
        ax1.set_ylabel('pixels comprising billboard', color=color)
        ax1.axvline(x=18, linewidth=5, alpha=0.3)
        ax1.axvline(x=20, linewidth=5, alpha=0.3)
        ax1.axvline(x=22, linewidth=5, alpha=0.3)
        ax1.axvline(x=24, linewidth=5, alpha=0.3)
        ax1.axvline(x=28, linewidth=5, alpha=0.3)
        try:
            ax1.set_xticks(np.arange(math.floor(min(distances)), math.ceil(max(distances)) + 1, 1.0))
        except:
            pass
        lns1 = ax1.plot(distances, errors, color=color)
        ax1.invert_xaxis()

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.title.set_text(title)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig("{}/{}-{}-inital_cuton_angle_error.jpg".format(new_results_dir, default_scenario, default_spawnpoint))
        plt.close("all")


def plot_all_errors_by_cuton(ds, errs, title):
    global newdir, new_results_dir
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('distance to billboard (M)')
    ax1.set_ylabel('angle error', color=color)
    ax1.axvline(x=18, linewidth=5, alpha=0.3)
    ax1.axvline(x=20, linewidth=5, alpha=0.3)
    ax1.axvline(x=22, linewidth=5, alpha=0.3)
    ax1.axvline(x=24, linewidth=5, alpha=0.3)
    ax1.axvline(x=28, linewidth=5, alpha=0.3)
    for distances,errors in zip(ds, errs):
        lns1 = ax1.plot(distances, errors, color=color)
    ax1.invert_xaxis()
    # lns2 = ax1.plot(detected_runtimes, detected_percents, label="ratio w/ billboard detected", color='tab:orange')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.title.set_text(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("{}/{}-{}-all_cuton_angle_error.jpg".format(new_results_dir, default_scenario, default_spawnpoint))
    plt.close("all")


def run_scenario_to_collect_sequence(vehicle, bng, model, spawn, direction="left", run_number=0, device=torch.device('cuda'),
                                     outdir=None, bb_size=None, iterations=None, noise_level=None):
    global base_filename, steps_per_sec, expected_trajectory, unperturbed_steer
    global integral, prev_error, setpoint
    print("run_scenario_to_collect_sequence")
    bng.restart_scenario()
    bng.pause()
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)
    qrbox_pos = list(qr_positions[0][0])
    integral = 0.0;
    prev_error = setpoint;
    runtime = 0.0
    damage = sensors['damage']['damage'];
    final_img = None
    kphs = [];
    traj = [];
    ys = []
    all_ys = []
    start_time = sensors['timer']['time']
    imagecount = 0
    sequence = []
    steering_inputs = []
    steering_inputs_pert = []
    steering_inputs_orig = []
    distances = []
    outcome = None
    calls = 0
    pixel_sizes = []
    while damage <= 0:
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_cam']['colour'].convert('RGB')
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        runtime = sensors['timer']['time'] - start_time
        damage = sensors['damage']['damage']
        colorseg_img = sensors['front_cam']['annotation'].convert('RGB')
        qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
        print(f"{qr_corners=}")
        bb_percent, pixelsize = get_percent_of_image(qr_corners, bbox_img)
        print(f"{pixelsize=}")
        pixel_sizes.append(pixelsize)
        # qr_corners, bbox_img = get_qr_corners_from_colorseg_image_nowarp(colorseg_img)
        cv2.imshow('car view', np.array(image)[:, :, ::-1])
        cv2.waitKey(1)
        # cv2.imwrite("results/SDBB-imgcoll/{}.jpg".format(len(sequence)), np.array(image)[:, :, ::-1])
        dist_to_bb = distance(vehicle.state['pos'], qrbox_pos)
        distances.append(dist_to_bb)
        print(f"{dist_to_bb=}\n")
        if bbox_img is not None and kph > 29:
            sequence.append({"image": model.process_image(image)[0], "bbox": qr_corners})
            slice = [{"image": model.process_image(image)[0], "bbox": qr_corners}]
            imagecount += 1
            if not is_billboard_fully_viewable(image, qr_corners):
                print("Billboard no longer viewable")
                outcome = "R2NT"
                break

        with torch.no_grad():
            # model = model.cpu()
            prediction = model(model.process_image(image).to(device))
        if bbox_img is not None and kph > 29 and outdir is not None:
            calls += 1
            # pert_billboard, pert_ys, perturbed_imgs, arrowed_imgs = gradientascent(model, slice, direction, bb_size=bb_size, iterations=iterations,
            #                                noise_level=noise_level, outdir=outdir)
            # distances.append(dist_to_bb)
            steering_inputs_orig.append(prediction)
            # steering_inputs_pert.append(pert_ys[-1])

        if bbox_img is not None and kph > 29:
            ys.append(prediction)

        # control params
        dt = (sensors['timer']['time'] - start_time) - runtime
        steering = float(prediction[0][0])
        steering_inputs.append(steering)
        if abs(steering) > 0.2:
            setpoint = 30
        else:
            setpoint = 40
        throttle = throttle_PID(kph, dt)
        vehicle.control(throttle=throttle, steering=steering, brake=0.0)

        # collect metrics
        traj.append(vehicle.state['pos'])
        kphs.append(kph)
        final_img = image
        all_ys.append(steering)

        if damage > 0.0:
            print("Damage={}, exiting...".format(damage))
            outcome = "D={}".format(round(damage,2))
            break
        if has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng):
            print("Left track, exiting...")
            outcome = "LT"
            break
        # if dist_to_bb < 25:
        #     print("EARLY BREAK")
        #     outcome = 'LT'
        #     break
        bng.step(1, wait=True)
    unperturbed_steer = steering_inputs
    cv2.destroyAllWindows()
    # plt.title("Run-{}-finalimg".format(run_number))
    # plt.imshow(final_img)
    # plt.pause(0.01)
    print(f"Sequence collected; {imagecount=}\n")
    deviation, dists, avg_dist = calc_deviation_from_center(expected_trajectory, traj)
    results = {'runtime': round(runtime, 3), 'damage': damage, 'kphs': kphs, 'traj': traj, 'final_img': final_img,
               'deviation': deviation, 'dists': dists, 'avg_dist': avg_dist, 'ys': ys, "outcome": outcome, "all_ys" : all_ys #,
               # 'distances':distances, 'inputs_orig':steering_inputs_orig, 'inputs_pert':steering_inputs_pert, 'errors': np.array(steering_inputs_orig) - np.array(steering_inputs_pert)
               }
    plot_errors_by_cuton(distances, pixel_sizes, f"Pixel size of billboard by distance\nbbsize={bb_size} iters={iterations} noise_variance=1/{noise_level}")
    return sequence, results


def main():
    global base_filename, default_scenario, default_spawnpoint, setpoint, integral
    global prev_error, centerline, centerline_interpolated, unperturbed_traj, new_results_dir
    start = time.time()
    setup_args = get_sequence_setup(default_spawnpoint)
    direction = "left"
    techniques = ["dbb"]
    # model_name = "model-DAVE2v3-lr1e4-50epoch-batch64-lossMSE-25Ksamples.pt"
    # model_name = "model-DAVE2PytorchModel-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    model_name = "model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    lossname = ''
    bbsizes = [5]
    iterations = [400]
    noiselevels = [5, 10, 15]

    vehicle, bng, model, spawn = setup_beamng(setup_args, vehicle_model='hopper', model_name=model_name)

    newdir = "SANITYCHECK-imagebyimage-DNNnew{}model-{}".format(model._get_name(), ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)))
    if not os.path.isdir("results/{}".format(newdir)):
        os.mkdir("results/{}".format(newdir))

    intake_lap_file(f"DAVE2v1-lap-trajectory.txt")
    centerline, left, right = road_analysis(bng)
    create_ai_line_from_road_with_interpolation()

    ds = []
    errs = []
    for bbsize in bbsizes:
        for its in iterations:
            for nl in noiselevels:
                for technique in techniques:

                    localtime = time.localtime()
                    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
                    new_results_dir = "results/{}/results-{}-{}-{}-{}-{}-{}".format(newdir, technique, bbsize, nl, its, timestr, ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)))
                    if not os.path.isdir(new_results_dir):
                        os.mkdir(new_results_dir)
                    training_file = "{}/{}-{}-{}-{}-{}-bbsize{}-{}iters-noisedenom{}.pickle".format(new_results_dir, technique,
                                                                                                         model._get_name(),
                                                                                                         technique, direction,
                                                                                                         lossname, bbsize, its, nl)

                    lossname = 'MAE'
                    results = run_scenario(vehicle, bng, model, spawn, direction, collect_sequence_results={},
                                                              bb_size=bbsize, iterations=its, noise_level=nl, resultsdir=new_results_dir)
                    ds.append(results['distances'])
                    errs.append(results['errors'])
    plot_all_errors_by_cuton(ds, errs, title='All angle errors by distance to billboard')
    bng.close()
    # except:
    #     bng.close()
    print(f"Finished in {time.time() - start} seconds")


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    main()
