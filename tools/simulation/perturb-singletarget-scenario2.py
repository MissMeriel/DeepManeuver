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

from perturbation_generator import DeepBillboard, DeepManeuver
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
from sim_utils import *
# globals
default_color = 'White'
default_scenario = "industrial"
default_spawnpoint = "scenario2"
direction = "left"
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40.0
setpoint = overall_throttle_setpoint
lanewidth = 3.75  # 2.25
centerline, centerline_interpolated = [], []
roadmiddle, roadleft, roadright = [], [], []
expected_trajectory, unperturbed_traj, unperturbed_steer = [], [], []
steps_per_sec = 15
newdir, new_results_dir = '', ''
qr_positions = []
unperturbed_seq = None


def throttle_PID(kph, dt):
    global integral, prev_error, setpoint
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


def road_analysis(bng):
    global centerline, roadleft, roadright, roadmiddle
    global default_scenario, default_spawnpoint
    # plot_racetrack_roads(bng.get_roads(), bng, default_scenario, default_spawnpoint)
    # get relevant road
    edges = []
    adjustment_factor = 4.0
    if default_scenario == "industrial" and default_spawnpoint == "racetrackstartinggate":
        edges = bng.get_road_edges('7982')
    elif default_scenario == "industrial" and (default_spawnpoint == "scenario1" or  default_spawnpoint == "scenario2"):
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
    # print(f"{args.road_id=}")
    # print(f"{len(actual_middle)=}")
    roadmiddle = copy.deepcopy(actual_middle)
    # print(f"{actual_middle[0]=}")
    roadleft = [edge['left'] for edge in edges]
    roadright = [edge['right'] for edge in edges]
    adjusted_middle = [np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle']))/adjustment_factor for edge in edges]
    centerline = actual_middle
    return actual_middle, adjusted_middle, roadleft, roadright


def create_ai_line_from_road_with_interpolation(spawn, bng):
    global centerline, remaining_centerline, centerline_interpolated, roadleft, roadright
    points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
    actual_middle, adjusted_middle, roadleft, roadright = road_analysis(bng)
    print("finished road analysis")
    start_index = get_start_index(adjusted_middle, default_scenario, default_spawnpoint)
    middle_end = adjusted_middle[:start_index]
    middle = adjusted_middle[start_index:]
    # temp = [list(spawn['pos'])]; temp.extend(middle); middle = temp
    middle.extend(middle_end)
    middle.append(middle[0])
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
    print("spawn point:{}".format(spawn))
    print("beginning of script:{}".format(middle[0]))
    # plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
    centerline = copy.deepcopy(traj)
    remaining_centerline = copy.deepcopy(traj)
    centerline_interpolated = copy.deepcopy(traj)
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return bng


def setup_beamng(vehicle_model='hopper',
                 model_name="test-7-trad-50epochs-64batch-1e4lr-ORIGDATASET-singleoutput-model-epoch-43.pt"):
    global default_scenario, default_spawnpoint
    global new_results_dir, default_color, steps_per_sec, qr_positions
    global integral, prev_error, setpoint
    integral = 0.0
    prev_error = 0.0
    model = torch.load(f"{args.path2src}/GitHub/superdeepbillboard/models/{model_name}", map_location=torch.device('cuda')).eval()
    setup_logging()

    beamng = BeamNGpy('localhost', 64256, home=f'{args.path2src}/BeamNG.research.v1.7.0.1', user=f'{args.path2src}/BeamNG.research')
    scenario = Scenario(default_scenario, 'research_test')
    vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color=default_color)
    vehicle = setup_sensors(vehicle)
    spawn = spawn_point(default_scenario, default_spawnpoint)
    scenario.add_vehicle(vehicle, pos=spawn['pos'], rot=None, rot_quat=spawn['rot_quat'])
    scenario = add_barriers(scenario)
    qr_positions = add_qr_cubes(scenario, default_scenario, default_spawnpoint)

    scenario.make(beamng) # Compile the scenario and place it in BeamNG's map folder
    bng = beamng.open(launch=True) # Start BeamNG and enter the main loop
    bng.set_steps_per_second(steps_per_sec)  # With 36hz temporal resolution
    bng.set_deterministic()  # Set simulator to be deterministic
    # bng.set_particles_enabled
    bng.load_scenario(scenario)
    bng.start_scenario()
    bng.pause() # Put simulator in pause awaiting further inputs
    assert vehicle.skt
    return vehicle, bng, model, spawn


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


def deepmaneuver(model, sequence, direction, steering_vector, bb_size=5, iterations=400, noise_level=25,
                       dist_to_bb=None, last_billboard=None, input_divers=True, loss_fxn='inv23'):
    sdbb = DeepManeuver.DeepManeuver(model, sequence, direction)
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
        constraint = 0.33 #dist_to_bb / 22.0 #math.atan(dist_to_bb / 22.0) # 0.2 # 0.1667
        if direction == "left":
            steering_vector.append(-constraint)  # steering_vector[-1]-0.1
        else:
            steering_vector.append(constraint)  # steering_vector[-1]+0.1
    else:
        constraint = 0.33 # 0.2  # 0.1667
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
    global centerline, default_spawnpoint, unperturbed_traj, unperturbed_steer, unperturbed_seq
    starttime = time.time()
    sequence, unperturbed_results = run_scenario_to_collect_sequence(vehicle, bng, model, spawn, cuton=dist_to_bb_cuton)
    sequence.extend(sequence)
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

def run_scenario_deepmaneuver(vehicle, bng, model, spawn, direction, dist_to_bb_cuton=26, dist_to_bb_cutoff=0, run_number=0,
                                    collect_sequence_results=None,
                                    bb_size=5, iterations=400, noise_level=25, resultsdir="images", input_divers=True, loss_fxn='inv23'):
    global default_spawnpoint, unperturbed_traj, unperturbed_steer
    starttime = time.time()
    pert_billboards, perturbation_run_results = run_scenario_for_deepmaneuver(vehicle, bng, model, spawn, direction,
                                                                                bb_size=bb_size, iterations=iterations, noise_level=noise_level,
                                                                                dist_to_bb_cuton=dist_to_bb_cuton, dist_to_bb_cutoff=dist_to_bb_cutoff,
                                                                                input_divers=input_divers, loss_fxn=loss_fxn)
    timetorun = time.time() - starttime
    print("Time to perturb:", timetorun)
    plt.title("sdbb final pert_billboard")
    plt.imshow(pert_billboards[-1])
    plt.savefig("{}/pert_billboard-sdbb-{}-{}-{}.jpg".format(resultsdir, bb_size, iterations, noise_level))
    plt.close('all')
    pert_trajs, Ys = [], []
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


def run_scenario_to_collect_sequence(vehicle, bng, model, spawn, cuton=40, device=torch.device('cuda')):
    global new_results_dir, steps_per_sec, expected_trajectory, unperturbed_steer, roadmiddle
    global integral, prev_error, setpoint, unperturbed_seq
    print("run_scenario_to_collect_sequence")
    bng.restart_scenario()
    bng.pause()
    vehicle.update_vehicle()
    sensors = bng.poll_sensors(vehicle)

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
    unperturbed_seq = []

    while damage <= 0:
        vehicle.update_vehicle()
        sensors = bng.poll_sensors(vehicle)
        image = sensors['front_cam']['colour'].convert('RGB')
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        runtime = sensors['timer']['time'] - start_time
        damage = sensors['damage']['damage']
        colorseg_img = sensors['front_cam']['annotation'].convert('RGB')
        qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
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
            unperturbed_seq.append({"image": model.process_image(image)[0], "bbox": qr_corners})
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
        else:
            setpoint = overall_throttle_setpoint
        throttle = throttle_PID(kph, dt)
        vehicle.control(throttle=throttle, steering=steering, brake=0.0)

        # collect metrics
        traj.append(vehicle.state['front'])
        kphs.append(kph)
        final_img = image
        all_ys.append(steering)

        rs = calc_points_of_reachable_set(vehicle.state)
        road_seg = nearest_seg(roadmiddle, vehicle.state['front'], roadleft, roadright)
        x = intersection_of_RS_and_road(rs, road_seg)
        # plot_intersection_with_CV2(vehicle.state, rs, road_seg, x, vehicle.get_bbox())

        if damage > 0.0:
            print(f"Damage={damage:.3f}, exiting...")
            outcome = "D={}".format(round(damage,2))
            break
        if has_car_left_track(vehicle.state['pos'], centerline_interpolated):
            print("Left track, exiting...")
            outcome = "LT"
            break

        bng.step(1, wait=True)
        last_steering_from_sim = sensors['electrics']['steering_input']
    unperturbed_steer = steering_inputs
    cv2.destroyAllWindows()
    plot_billboard_ratios(runtimes, percents, detected_runtimes, detected_percents, distances, detected_distances,
                          "Billboard Ratio of Image during Normal Run (No Perturbation)", newdir, new_results_dir, default_scenario, default_spawnpoint)

    print(f"Sequence collected; {len(unperturbed_seq)=}\n")
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
    pos_window = np.zeros((10, 3))
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
        elif has_car_left_track(vehicle.state['pos'], centerline_interpolated):
            outcomestring = f"LT"
            print("Left track, exiting...")
            break
        elif not billboard_viewable and returned_to_expected_traj(pos_window, expected_trajectory):
            outcomestring = "R2NT"
            print("Returned to normal trajectory, exiting...")
            break
        elif distance(spawn['pos'], vehicle.state['pos']) > 65 and runtime > 10:
            outcomestring = "2FAR"
            print("Too far from sequence, exiting...")
            break
        bng.step(1, wait=True)

    cv2.destroyAllWindows()
    plot_MAEs(distances, percents, detected_distances, detected_percents, unperturbed_predictions,
               perturbed_predictions, angleerror_distances, new_results_dir=new_results_dir,
               title=f"Angle Error during Test Run w.r.t. Distance\ncuton={dist_to_bb_cuton}M cutoff={dist_to_bb_cutoff}")
    mse = mean_squared_error(unperturbed_predictions, perturbed_predictions)
    error = np.array(unperturbed_predictions) - np.array(perturbed_predictions)
    deviation, dists, avg_dist = calc_deviation_from_center(expected_trajectory, traj)
    results = {'runtime': round(runtime, 3), 'damage': damage, 'kphs': kphs, 'traj': traj, 'final_img': final_img,
               'deviation': deviation, 'mse': mse, 'dists': dists, 'avg_dist': avg_dist, 'error': error,
               'perturbed_predictions': perturbed_predictions, 'outcome': outcomestring, "all_ys":all_ys
               }
    return results


def run_scenario_for_deepmaneuver(vehicle, bng, model, spawn, direction, dist_to_bb_cuton=26, dist_to_bb_cutoff=26,
                                        bb_size=5, iterations=100, noise_level=25, input_divers=True, loss_fxn='inv23',
                                        device=torch.device('cuda')):
    global default_spawnpoint, unperturbed_traj, unperturbed_steer, unperturbed_seq
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
    # if add_exp_traj:
    #     for i in unperturbed_seq:
    #         sequence.append(i)
    #         steering_vector.append(1.0)
    ys, all_ys = [], []
    bb_viewed_window = np.ones((10))
    detected_runtimes, runtimes, angleerror_runtimes = [], [], []
    detected_percents, percents = [], []
    detected_distances, distances = [], []
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
            almost_left_track = has_car_almost_left_track(vehicle.state['front'], centerline_interpolated)
            rs = calc_points_of_reachable_set(vehicle.state)
            road_seg = nearest_seg(roadmiddle, vehicle.state['front'], roadleft, roadright)
            x = intersection_of_RS_and_road(rs, road_seg)
            x = round(x,2)
            print(f"RS overlap={x}")
            plot_intersection_with_CV2(vehicle.state, rs, road_seg, x, vehicle.get_bbox())

            # stopping conditions
            if damage > 0.0:
                print("Damage={} at timestep={}, exiting...".format(damage, round(runtime, 2)))
                outcome = "D={}".format(round(damage, 2))
                break

            # if kph > 29 and has_car_almost_left_track(vehicle.state['front'], centerline_interpolated) and dist_to_bb < dist_to_bb_cuton:
            #     print("Almost left track, exiting...")
            #     outcome = "LT"
            #     break

            if (kph > 29 and abs(x - dist_to_bb_cutoff) <= 0.02) or (kph > 29 and x <= dist_to_bb_cutoff):
                print(f"RS overlap={x}, exiting...")
                outcome = f"RS overlap={x}"
                break

        if bbox_img is not None and kph > 29 and round(dist_to_bb,0) <= dist_to_bb_cuton: # and (steps % 2 < 1):
            sequence.append({"image": model.process_image(image)[0], "bbox": qr_corners, "colorseg_img":model.process_image(colorseg_img)[0]})
            sequence2 = copy.deepcopy(sequence)
            steering_vector2 = copy.deepcopy(steering_vector)
            for i in unperturbed_seq:
                sequence2.append(i)
                steering_vector2.append(1)

            # sequence2 = []
            # steering_vector2 = []
            # for i in range(len(sequence)-1):
            #     sequence2.append(sequence[i])
            #     sequence2.append(unperturbed_seq[i])
            #     steering_vector2.append(steering_vector[i])
            #     steering_vector2.append(1)
            # sequence2.append(sequence[-1])
            # sequence2.append(unperturbed_seq[len(sequence)-1])
            # steering_vector2.append(1)

            pert_billboard, y, MAE = deepmaneuver(model, sequence2, direction, copy.deepcopy(steering_vector2),
                                                   bb_size=bb_size, iterations=iterations, noise_level=noise_level,
                                                   dist_to_bb=dist_to_bb, input_divers=input_divers, loss_fxn=loss_fxn)
            steering_vector.append(last_steering_from_sim)
            model = model.to(device)
            ys = y
            pert_billboards.append(copy.deepcopy(pert_billboard))
            image = add_perturbed_billboard(origimage, pert_billboard, qr_corners[0])
            cv2.imshow('car view', image[:, :, ::-1])
            cv2.waitKey(1)
        # elif bbox_img is not None and kph > 29:
        #     sequence.append({"image": model.process_image(image)[0], "bbox": qr_corners,
        #                      "colorseg_img": model.process_image(colorseg_img)[0]})
        #     steering_vector.append(last_steering_from_sim)

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


def main():
    global new_results_dir, newdir, default_scenario, default_spawnpoint, setpoint, integral
    global prev_error, centerline, centerline_interpolated, unperturbed_traj, direction
    global expected_trajectory, steps_per_sec
    start = time.time()
    direction = "right"
    techniques = ["dbb-orig", "dbb-plus", "sdbb"]
    model_name = "model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
    lossname, new_results_dir = '', ''
    bbsizes = [15]
    iterations = [400]
    noiselevels = [15]
    rscs = [0.60]
    cutons = [28]
    input_divs = [False]
    vehicle, bng, model, spawn = setup_beamng(vehicle_model='hopper', model_name=model_name)
    steps_per_sec = 15
    samples = 2
    newdir = "TSErev-TABLE1TOPO11-{}".format(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)))
    if not os.path.isdir("results/{}".format(newdir)):
        os.mkdir("results/{}".format(newdir))
        print(f"Copying script to {os.getcwd()}/{__file__}")
        shutil.copy(f"{__file__}", f"results/{newdir}")
        shutil.copy(f"{os.getcwd()}/../perturbation_generator/DeepManeuver.py", f"results/{newdir}")
    expected_trajectory = intake_lap_file(f"posefiles/DAVE2v3-lap-trajectory.txt")
    bng = create_ai_line_from_road_with_interpolation(spawn, bng)

    sequence, unperturbed_results = run_scenario_to_collect_sequence(vehicle, bng, model, spawn, 28)
    unperturbed_traj = unperturbed_results['traj']
    for cuton in cutons:
        for rsc in rscs:
            for bbsize in bbsizes:
                for its in iterations:
                    for nl in noiselevels:
                        for technique in techniques:
                            for input_div in input_divs:
                                for i in range(samples):
                                    all_trajs, all_outcomes = [], []
                                    localtime = time.localtime()
                                    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)

                                    if technique == "dbb":
                                        lossname = 'MDirE'
                                        new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, nl, its, cuton, rsc, timestr)
                                        results = run_scenario(vehicle, bng, model, spawn, direction, collect_sequence_results=unperturbed_results,
                                                                                  bb_size=bbsize, iterations=its, noise_level=nl, dist_to_bb_cuton=cuton,
                                                                                    resultsdir=new_results_dir)
                                    elif technique == "dbb-orig":
                                        lossname = 'MDirE'
                                        new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, 1000,
                                                                                           its, 28, rsc, input_div=False, timestr=timestr)
                                        results = run_scenario(vehicle, bng, model, spawn, direction,
                                                               collect_sequence_results=unperturbed_results,
                                                               bb_size=bbsize, iterations=its, noise_level=1000,
                                                               dist_to_bb_cuton=28,
                                                               resultsdir=new_results_dir, input_divers=False)
                                    elif technique == "dbb-plus":
                                        lossname = 'MDirE'
                                        new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, nl,
                                                                                           its, cuton, rsc, input_div, timestr)
                                        results = run_scenario(vehicle, bng, model, spawn, direction,
                                                               collect_sequence_results=unperturbed_results,
                                                               bb_size=bbsize, iterations=its, noise_level=nl,
                                                               dist_to_bb_cuton=cuton, input_divers=input_div, resultsdir=new_results_dir)
                                    else:
                                        lossname = 'inv23' # 'MDirE'
                                        new_results_dir, training_file = make_results_dirs(newdir, technique, bbsize, nl,
                                                                                           its, cuton, rsc, input_div, timestr)
                                        results = run_scenario_deepmaneuver(vehicle, bng, model, spawn, direction, collect_sequence_results=unperturbed_results,
                                                                                  bb_size=bbsize, iterations=its, noise_level=nl, dist_to_bb_cuton=cuton,
                                                                                  dist_to_bb_cutoff=rsc, resultsdir=new_results_dir, input_divers=input_div, loss_fxn=lossname)
                                        all_trajs.append(results["pertrun_traj"])
                                        all_outcomes.append(results["pertrun_outcome"])

                                    cv2.destroyAllWindows()
                                    all_trajs.extend(results['testruns_trajs'])
                                    all_outcomes.extend(results['testruns_outcomes'])

                                    write_results(training_file, results, all_trajs, unperturbed_traj,
                                                  model._get_name(), technique, direction, lossname, bbsize, its, nl)

                                    plot_deviation(all_trajs, unperturbed_traj,
                                                   "{}-{}-{}\nDOF{}-noisevar{}-cuton{}".format(technique, direction, lossname, bbsize, nl, cuton),
                                                   centerline_interpolated, roadleft, roadright, all_outcomes, resultsdir=new_results_dir,
                                                   qr_positions=qr_positions, default_scenario=default_scenario, default_spawnpoint=default_spawnpoint)
                                    plot_deviation(all_trajs, unperturbed_traj,
                                                   "{}-{}-{}\nDOF{}-noisevar{}-cuton{}-ZOOMED".format(technique, direction, lossname, bbsize, nl, cuton),
                                                   centerline_interpolated, roadleft, roadright, all_outcomes, resultsdir=new_results_dir,
                                                   qr_positions=qr_positions, default_scenario=default_scenario, default_spawnpoint=default_spawnpoint)
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
