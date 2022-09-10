import argparse

parser = argparse.ArgumentParser(description="Process paths")
parser.add_argument(
    "path2src", metavar="N", type=str, help="path to source parent dirs"
)
parser.add_argument("road_id", metavar="N", type=str, help="road identifier in BeamNG")
args = parser.parse_args()
print(args)

import warnings
from functools import wraps
import logging, random, string, time, copy, os, sys, shutil
import math
import pickle
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

sys.path.append(f"{args.path2src}/GitHub/BeamNGpy")
sys.path.append(f"{args.path2src}/GitHub/BeamNGpy/src/")
print(sys.path)

from perturbationgenerator import DeepBillboard, SuperDeepBillboard

# from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
# from beamngpy import ProceduralCube #,ProceduralCylinder, ProceduralCone, ProceduralBump, ProceduralRing
# from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
# from scipy.spatial.transform import Rotation as R
# from ast import literal_eval
# from scipy import interpolate
# from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import torch
import cv2
from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage import util
import kornia
from torchvision.utils import save_image
from shapely.geometry import Polygon

# project imports
from simulator import Simulator
from models.DAVE2pytorch import DAVE2v3

# globals
default_scenario = "industrial"
default_spawnpoint = "straight1"
integral, prev_error = 0.0, 0.0
overall_throttle_setpoint = 40.0
setpoint = overall_throttle_setpoint
unperturbed_traj, unperturbed_steer = [], []
newdir, new_results_dir = "", ""
qr_positions = []
unperturbed_seq = None

# def ignore_warnings(f):
#     @wraps(f)
#     def inner(*args, **kwargs):
#         with warnings.catch_warnings(record=True) as w:
#             warnings.simplefilter("ignore")
#             response = f(*args, **kwargs)
#         return response
#     return inner


def get_outcomes(results):
    outcomes_counts = {"D": 0, "LT": 0, "R2NT": 0, "2FAR": 0}
    # print(results.keys())
    total = float(len(results["testruns_outcomes"]) - 1)
    for outcome in results["testruns_outcomes"]:
        if "D=" in outcome:
            outcomes_counts["D"] += 1
        elif "R2NT" in outcome:
            outcomes_counts["R2NT"] += 1
        elif "LT" in outcome:
            outcomes_counts["LT"] += 1
        elif "2FAR" in outcome:
            outcomes_counts["2FAR"] += 1
    return outcomes_counts


def ms_to_kph(wheelspeed):
    return wheelspeed * 3.6


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


def plot_deviation(trajectories, unperturbed_traj, model, centerline, left, right, outcomes, resultsdir="images"):
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
    plt.plot(x, y, "k")
    x, y = [], []
    for point in left:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k")
    x, y = [], []
    for point in right:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k")
    x, y = [], []
    for i, t in enumerate(zip(trajectories, outcomes)):
        x = [point[0] for point in t[0]]
        y = [point[1] for point in t[0]]
        if i == 0 and "sdbb" in model:
            plt.plot(x, y, label="Pert. Run {} ({})".format(i, t[1]), linewidth=5)
        else:
            plt.plot(x, y, label="Pert. Run {} ({})".format(i, t[1]), alpha=0.75)
        i += 1
    plt.plot(
        [p[0][0] for p in qr_positions],
        [p[0][1] for p in qr_positions],
        "r", linewidth=5,
    )
    failures = 0
    for o in outcomes:
        if o == "LT" or "D" in o:
            failures += 1
    plt.title(
        "Trajectories with {} failures={}".format(model, failures),
        fontdict={"fontsize": 10},
    )
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
    elif ( default_scenario == "driver_training" and default_spawnpoint == "approachingfork"):
        plt.xlim([-50, 55])
        plt.ylim([150, 255])
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    randstr = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    plt.savefig("{}/{}-{}.jpg".format(resultsdir, model.replace("\n", "-"), randstr))
    plt.close("all")
    del x, y

# @ignore_warnings
def write_results(training_file, results, all_trajs, unperturbed_traj, modelname, technique, direction, lossname, bbsize, its, nl):
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


# return distance between two 3d points
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def calc_points_of_reachable_set(vehicle_state):
    turn_rad = math.radians(30)
    offset = math.radians(90)
    yaw = vehicle_state["yaw"][0]
    points = []
    radius = 11.1
    # leftmost point
    points.append(
        [
            vehicle_state["front"][0] + radius * math.cos(yaw + turn_rad - offset),
            vehicle_state["front"][1] + radius * math.sin(yaw + turn_rad - offset),
            vehicle_state["front"][2],
        ]
    )
    # front point touching car
    points.append(vehicle_state["front"])
    # rightmost point
    points.append(
        [
            vehicle_state["front"][0] + radius * math.cos(yaw - turn_rad - offset),
            vehicle_state["front"][1] + radius * math.sin(yaw - turn_rad - offset),
            vehicle_state["front"][2],
        ]
    )
    return points


def intersection_of_RS_and_road(rs, road_seg):
    segpts = copy.deepcopy(road_seg["left"])
    temp = road_seg["right"]
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


# @ignore_warnings
def plot_intersection_with_CV2(vehicle_state, rs, road_seg, intersection, bbox, traj=None):
    global qr_positions
    fig, ax = plt.subplots()
    yaw = vehicle_state["yaw"][0]
    patches = []
    radius = 11.1
    # plot LR limits of RS
    plt.plot(
        [p[0] for p in rs],
        [p[1] for p in rs],
        "tab:purple",
        label="reachable set (1 sec.)",
    )
    if traj is not None:
        plt.plot(
            [p[0] for p in traj], [p[1] for p in traj], "tab:purple", label="trajectory"
        )
    # plot area of RS
    wedge = Wedge(
        (vehicle_state["front"][0], +vehicle_state["front"][1]),
        radius,
        math.degrees(yaw) - 30 - 90,
        math.degrees(yaw) + 30 - 90,
    )
    patches.append(wedge)
    p = PatchCollection(patches, alpha=0.4)
    colors = np.array([70.0])
    p.set_array(colors)
    ax.add_collection(p)
    x = np.array([bbox[k][0] for k in bbox.keys()])
    y = np.array([bbox[k][1] for k in bbox.keys()])
    plt.plot(x, y, "m", label="car (bounding box)")
    plt.plot([x[0], x[-1]], [y[0], y[-1]], "m", linewidth=1)
    plt.plot([x[4], x[2], x[1], x[-1]], [y[4], y[2], y[1], y[-1]], "m", linewidth=1)
    # plt.plot([vehicle_state['front'][0]], [vehicle_state['front'][1]], "ms", label="car (front)")
    # add road segment
    for k in road_seg.keys():
        plt.plot(
            [road_seg[k][i][0] for i in range(len(road_seg[k]))],
            [road_seg[k][i][1] for i in range(len(road_seg[k]))],
            "k",
        )
    # add billboard
    plt.plot(
        [p[0][0] for p in qr_positions],
        [p[0][1] for p in qr_positions],
        linewidth=5,
        label="billboard",
    )
    plt.title(f"Reachable Set Intersection ({intersection*100:.2f}%)")
    plt.legend()
    plt.axis("square")
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("plot", img)
    cv2.waitKey(1)
    plt.close("all")


def plot_intersection(vehicle_state, rs, road_seg, intersection, bbox):
    fig, ax = plt.subplots()
    yaw = vehicle_state["yaw"][0]
    patches = []
    radius = 11.1
    # plot LR limits of RS
    plt.plot(
        [p[0] for p in rs],
        [p[1] for p in rs],
        "tab:purple",
        label="reachable set (1 sec.)",
    )
    # plot area of RS
    wedge = Wedge(
        (vehicle_state["front"][0], +vehicle_state["front"][1]),
        radius,
        math.degrees(yaw) - 30 - 90,
        math.degrees(yaw) + 30 - 90,
    )
    patches.append(wedge)
    p = PatchCollection(patches, alpha=0.4)
    colors = np.array([70.0])
    p.set_array(colors)
    ax.add_collection(p)
    x = np.array([bbox[k][0] for k in bbox.keys()])
    y = np.array([bbox[k][1] for k in bbox.keys()])
    plt.plot(x, y, "m", label="car (bounding box)")
    plt.plot([x[0], x[-1]], [y[0], y[-1]], "m", linewidth=1)
    plt.plot([x[4], x[2], x[1], x[-1]], [y[4], y[2], y[1], y[-1]], "m", linewidth=1)
    plt.plot(
        [vehicle_state["front"][0]],
        [vehicle_state["front"][1]],
        "ms",
        label="car (front)",
    )
    # add road segment
    for k in road_seg.keys():
        plt.plot(
            [road_seg[k][i][0] for i in range(len(road_seg[k]))],
            [road_seg[k][i][1] for i in range(len(road_seg[k]))],
            "k",
        )
    plt.title(f"Reachable Set Intersection ({intersection}%)")
    plt.legend()
    plt.close("all")


# with warp
def overlay_transparent(img1, img2, corners):
    orig = torch.from_numpy(img1)[None].permute(0, 3, 1, 2) / 255.0
    pert = torch.from_numpy(img2)[None].permute(0, 3, 1, 2) / 255.0

    _, c, h, w = _, *pert_shape = pert.shape
    _, *orig_shape = orig.shape
    patch_coords = corners[None]
    src_coords = np.tile(
        np.array([[[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]]]),
        (len(patch_coords), 1, 1),
    )
    src_coords = torch.from_numpy(src_coords).float()
    patch_coords = torch.from_numpy(patch_coords).float()

    # build the transforms to and from image patches
    try:
        perspective_transforms = kornia.geometry.transform.get_perspective_transform(
            src_coords, patch_coords
        )
    except Exception as e:
        print(f"{e=}")
        print(f"{src_coords=}")
        print(f"{patch_coords=}")

    perturbation_warp = kornia.geometry.transform.warp_perspective(
        pert,
        perspective_transforms,
        dsize=orig_shape[1:],
        mode="nearest",
        align_corners=True,
    )
    mask_patch = torch.ones(1, *pert_shape)
    warp_masks = kornia.geometry.transform.warp_perspective(
        mask_patch,
        perspective_transforms,
        dsize=orig_shape[1:],
        mode="nearest",
        align_corners=True,
    )
    perturbed_img = orig * (1 - warp_masks)
    perturbed_img += perturbation_warp * warp_masks
    return (perturbed_img.permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)


# uses contour detection
# @ignore_warnings
def get_qr_corners_from_colorseg_image(image):
    image = np.array(image)
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
    inverted_img = cv2.GaussianBlur(inverted_img, (3, 3), 0)  # 9

    # contour detection
    ret, thresh = cv2.threshold(inverted_img, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )

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
        for i, c in enumerate(le):
            cv2.circle(
                image,
                tuple([int(x) for x in c]),
                radius=1,
                color=(100 + i * 20, 0, 0),
                thickness=2,
            )  # blue
        for i, c in enumerate(re):
            cv2.circle(
                image,
                tuple([int(x) for x in c]),
                radius=1,
                color=(0, 0, 100 + i * 20),
                thickness=2,
            )  # blue
        cv2.circle(
            image, tuple(center), radius=1, color=(203, 192, 255), thickness=2
        )  # lite pink
        if len(approx) > 3:
            cv2.circle(
                image,
                tuple([int(x) for x in approx[0]]),
                radius=1,
                color=(0, 255, 0),
                thickness=2,
            )  # green
            cv2.circle(
                image,
                tuple([int(x) for x in approx[2]]),
                radius=1,
                color=(0, 0, 255),
                thickness=2,
            )  # red
            cv2.circle(
                image,
                tuple([int(x) for x in approx[3]]),
                radius=1,
                color=(255, 255, 255),
                thickness=2,
            )  # white
            cv2.circle(
                image,
                tuple([int(x) for x in approx[1]]),
                radius=1,
                color=(147, 20, 255),
                thickness=2,
            )  # pink

        keypoints = [
            [tuple(approx[0]), tuple(approx[3]), tuple(approx[1]), tuple(approx[2])]
        ]
        return keypoints, image


def is_billboard_fully_viewable(image, qr_corners):
    pixel_epsilon = 20
    imageheight = image.size[1]
    imagewidth = image.size[0]
    for corners in qr_corners:
        # ORDER: upper left, upper right, lower left, lower right (opencv image indices reversed)
        if (
            corners[0][0] <= pixel_epsilon
            or corners[0][1] <= pixel_epsilon
            or abs(imagewidth - corners[3][0]) <= pixel_epsilon
            or abs(imageheight - corners[3][1]) <= pixel_epsilon
        ):
            return False
    return True


def deepbillboard(
    model, sequence, direction, device=torch.device("cuda"), bb_size=5, iterations=400, noise_level=25, input_divers=False,
):
    deepbb = DeepBillboard.DeepBillboard(model, sequence, direction)
    img_arr = [hashmap["image"] for hashmap in sequence]
    img_patches = [hashmap["bbox"][0] for hashmap in sequence]
    new_img_patches = []
    for i, patch in enumerate(img_patches):
        temp = [i]
        for tup in patch:
            temp.append(tup[0])
            temp.append(tup[1])
        new_img_patches.append(copy.deepcopy(temp))
    perturbed_billboard_images = deepbb.perturb_images(img_arr, np.array(new_img_patches), model, device=device,
                                                        bb_size=bb_size, iterations=iterations, noise_level=noise_level,
                                                        input_divers=input_divers,
    )
    return perturbed_billboard_images


def get_percent_of_image(coords, img):
    coords = [tuple(i) for i in coords[0]]
    coords = tuple([coords[0], coords[1], coords[3], coords[2]])
    patch_size = Polygon(coords).area
    img_size = img.size[0] * img.size[1]
    return patch_size / img_size


def superdeepbillboard(
    model, sequence, direction, steering_vector, bb_size=5, iterations=400, noise_level=25,
    dist_to_bb=None, last_billboard=None, input_divers=True, loss_fxn="inv23", device=torch.device("cuda"),
):
    sdbb = SuperDeepBillboard.SuperDeepBillboard(model, sequence, direction)
    img_arr = [hashmap["image"] for hashmap in sequence]
    img_patches = [hashmap["bbox"][0] for hashmap in sequence]
    new_img_patches = []
    for i, patch in enumerate(img_patches):
        temp = [i]
        for tup in patch:
            temp.append(tup[0])
            temp.append(tup[1])
        new_img_patches.append(copy.deepcopy(temp))
    constraint = 1.0
    if direction == "left":
        steering_vector.append(-constraint)
    else:
        steering_vector.append(constraint)
    tensorized_steering_vector = torch.as_tensor(np.array(steering_vector, dtype=np.float64), dtype=torch.float)
    perturbed_billboard_images, y, MAE = sdbb.perturb_images(
        img_arr, np.array(new_img_patches), model, tensorized_steering_vector,
        device=device, bb_size=bb_size, iterations=iterations,
        noise_level=noise_level, last_billboard=last_billboard,
        loss_fxn=loss_fxn, input_divers=input_divers,
    )
    return perturbed_billboard_images, y, MAE


def run_scenario(sim, model, direction, bb_size=5, iterations=400, noise_level=25, dist_to_bb_cuton=37, resultsdir="images", input_divers=False):
    global default_spawnpoint, unperturbed_traj, unperturbed_steer, unperturbed_seq
    starttime = time.time()
    sequence, unperturbed_results = run_scenario_to_collect_sequence(sim, model, cuton=dist_to_bb_cuton)
    sequence.extend(sequence)
    pert_billboard, ys, MAE_collseq = deepbillboard(model, sequence, direction, bb_size=bb_size, iterations=iterations, noise_level=noise_level, input_divers=input_divers)
    timetorun = time.time() - starttime
    print(f"Time to perturb: {timetorun:.1f}")
    plt.title("dbb final pert_billboard")
    plt.imshow(pert_billboard)
    plt.savefig("{}/pert_billboard-dbb.jpg".format(resultsdir))
    plt.close("all")
    save_image(
        torch.from_numpy(pert_billboard).permute(2, 0, 1) / 255.0,
        "{}/dbb_pert_billboard_torchsaveimg.png".format(resultsdir),
    )
    pert_trajs = []
    Ys = []
    keys = ["unperturbed_deviation", "unperturbed_traj", "unperturbed_outcome", "testruns_deviation", "testruns_trajs",
            "testruns_dists", "testruns_ys", "testruns_error", "testruns_mse", "testruns_errors", "testruns_outcomes"
    ]
    values = [[] for k in keys]
    results = {key: value for key, value in zip(keys, values)}
    results["time_to_run_technique"] = timetorun
    results["unperturbed_outcome"] = unperturbed_results["outcome"]
    results["unperturbed_dists"] = unperturbed_results["dists"]
    results["unperturbed_deviation"] = unperturbed_results["deviation"]
    results["unperturbed_traj"] = unperturbed_results["traj"]
    results["unperturbed_all_ys"] = unperturbed_results["all_ys"]
    results["num_billboards"] = len(sequence)
    results["MAE_collection_sequence"] = MAE_collseq
    for i in range(10):
        runstarttime = time.time()
        perturbed_results = run_scenario_with_perturbed_billboard(sim, model, pert_billboard, run_number=i)
        print(f"Perturbed run {i} took {time.time()-runstarttime:2.2f}sec to finish.")
        results["testruns_deviation"].append(perturbed_results["deviation"])
        results["testruns_dists"].extend(perturbed_results["dists"])
        results["testruns_mse"].append(perturbed_results["mse"])
        results["testruns_error"].append(perturbed_results["error"])
        results["testruns_errors"].extend(perturbed_results["error"])
        results["testruns_outcomes"].append(perturbed_results["outcome"])
        Ys.append(perturbed_results["all_ys"])
        pert_trajs.append(perturbed_results["traj"])
        # results['errors'].extend(perturbed_results['error'])
    cv2.destroyAllWindows()
    results["testruns_trajs"] = pert_trajs
    results["testruns_all_ys"] = Ys
    results["unperturbed_deviation"] = unperturbed_results["deviation"]
    results["unperturbed_dists"] = unperturbed_results["dists"]
    results["pertrun_all_ys"] = None
    results["unperturbed_all_ys"] = unperturbed_results["all_ys"]
    outstring = (
        f"\nRESULTS FOR DBB {model._get_name()} {default_spawnpoint} {direction=} {bb_size=} {iterations=} {noise_level=}: \n"
        f"Avg. deviation from expected trajectory: \n"
        f"unperturbed:\t{results['unperturbed_deviation']}\n"
        f"perturbed:  \t{sum(results['testruns_deviation']) / float(len(results['testruns_deviation']))} \n"
        f"Avg. distance from expected trajectory:\n"
        f"unperturbed:\t{sum(results['unperturbed_dists']) / float(len(results['unperturbed_dists']))}\n"
        f"perturbed:  \t{sum(results['testruns_dists']) / float(len(results['testruns_dists']))}\n"
        f"Pred. angle error measures:\n"
        f"mse:      \t{sum(results['testruns_mse']) / float(len(results['testruns_mse']))}\n"
        f"avg error:\t{sum(results['testruns_errors']) / float(len(results['testruns_errors']))}\n"
        f"runtime:\t\t{timetorun}\n"
        f"num_billboards:\t\t{len(sequence)}\n"
        f"MAE:\t\t{MAE_collseq:.3f}"
    )
    print(outstring)
    return results


def run_scenario_superdeepbillboard(sim, model, direction, dist_to_bb_cuton=26, dist_to_bb_cutoff=0, device=torch.device("cuda"),
                                    collect_sequence_results=None, bb_size=5, iterations=400, noise_level=25, resultsdir="images",
                                    input_divers=True, loss_fxn="inv23",
):
    global default_spawnpoint, unperturbed_traj, unperturbed_steer
    starttime = time.time()
    pert_billboards, perturbation_run_results = run_scenario_for_superdeepbillboard(
        sim,
        model,
        direction,
        device=device,
        bb_size=bb_size,
        iterations=iterations,
        noise_level=noise_level,
        dist_to_bb_cuton=dist_to_bb_cuton,
        dist_to_bb_cutoff=dist_to_bb_cutoff,
        input_divers=input_divers,
        loss_fxn=loss_fxn,
    )
    timetorun = time.time() - starttime
    print("Time to perturb:", timetorun)
    plt.title("sdbb final pert_billboard")
    plt.imshow(pert_billboards[-1])
    plt.savefig(
        "{}/pert_billboard-sdbb-{}-{}-{}.jpg".format(
            resultsdir, bb_size, iterations, noise_level
        )
    )
    plt.close("all")
    pert_trajs, Ys = [], []
    keys = [
        "testruns_deviation",
        "testruns_dists",
        "testruns_ys",
        "testruns_mse",
        "testruns_error",
        "testruns_errors",
        "testruns_outcomes",
    ]
    values = [[] for k in keys]
    results = {key: value for key, value in zip(keys, values)}
    results["time_to_run_technique"] = timetorun
    results["unperturbed_dists"] = collect_sequence_results["dists"]
    results["unperturbed_deviation"] = collect_sequence_results["deviation"]
    results["unperturbed_traj"] = collect_sequence_results["traj"]
    results["unperturbed_all_ys"] = collect_sequence_results["all_ys"]
    results["pertrun_all_ys"] = perturbation_run_results["all_ys"]
    results["pertrun_outcome"] = perturbation_run_results["outcome"]
    results["pertrun_traj"] = perturbation_run_results["traj"]
    results["pertrun_deviation"] = perturbation_run_results["deviation"]
    results["pertrun_dist"] = perturbation_run_results["avg_dist"]
    results["num_billboards"] = perturbation_run_results["num_billboards"]
    results["MAE_collection_sequence"] = perturbation_run_results["MAE"]
    for i in range(10):
        # print(f"Run number {i}")
        runresults = run_scenario_with_perturbed_billboard(sim, model, pert_billboards[-1], dist_to_bb_cuton=dist_to_bb_cuton, dist_to_bb_cutoff=dist_to_bb_cutoff)
        results["testruns_deviation"].append(runresults["deviation"])
        results["testruns_dists"].extend(runresults["dists"])
        results["testruns_mse"].append(runresults["mse"])
        results["testruns_error"].append(runresults["error"])
        results["testruns_errors"].extend(runresults["error"])
        results["testruns_outcomes"].append(runresults["outcome"])
        Ys.append(runresults["all_ys"])
        pert_trajs.append(runresults["traj"])
    results["testruns_trajs"] = pert_trajs
    results["testruns_all_ys"] = Ys
    outstring = (
        f"\nRESULTS FOR SDBB {model._get_name()} {default_spawnpoint} {direction=} {bb_size=} {iterations=} {noise_level=}: \n"
        f"Avg. deviation from expected trajectory: \n"
        f"unperturbed:\t{results['unperturbed_deviation']}\n"
        f"pert.run: \t{results['pertrun_deviation']}\n"
        f"test:  \t\t{sum(results['testruns_deviation']) / float(len(results['testruns_deviation']))} \n"
        f"Avg. distance from expected trajectory:\n"
        f"unperturbed:\t{sum(results['unperturbed_dists']) / float(len(results['unperturbed_dists']))}\n"
        f"pert.run:\t\t{results['pertrun_dist']}\n"
        f"perturbed:  \t{sum(results['testruns_dists']) / float(len(results['testruns_dists']))}\n"
        f"Pred. angle error measures in test runs:\n"
        f"mse:      \t\t{sum(results['testruns_mse']) / float(len(results['testruns_mse']))}\n"
        f"avg error:\t{sum(results['testruns_errors']) / float(len(results['testruns_errors']))}\n"
        f"testruns outcomes: \t{results['testruns_outcomes']}\n"
        f"runtime:\t\t{timetorun}\n"
        f"MAE:\t\t{results['MAE_collection_sequence']:.3f}"
    )
    print(outstring)
    return results


def run_scenario_to_collect_sequence(sim, model, cuton=40, device=torch.device("cuda")):
    global unperturbed_steer, unperturbed_seq
    global integral, prev_error, setpoint
    print("run_scenario_to_collect_sequence")
    sim.restart()
    sensors = sim.get_sensor_readings()
    integral, runtime = 0.0, 0.0
    prev_error = setpoint
    damage = sensors["damage"]["damage"]
    start_time = sensors["timer"]["time"]
    final_img, outcome = None, None
    kphs, traj = [], []
    ys, all_ys = [], []
    imagecount = 0
    sequence, steering_inputs = [], []
    qrbox_pos = list(sim.qr_positions[0][0])
    percents, detected_percents = [], []
    detected_runtimes, runtimes = [], []
    distances, detected_distances = [], []
    unperturbed_seq = []
    while damage <= 0:
        sensors = sim.get_sensor_readings()
        vehicle_state = sim.get_vehicle_state()
        image = sensors["front_cam"]["colour"].convert("RGB")
        kph = ms_to_kph(sensors["electrics"]["wheelspeed"])
        runtime = sensors["timer"]["time"] - start_time
        damage = sensors["damage"]["damage"]
        colorseg_img = sensors["front_cam"]["annotation"].convert("RGB")
        qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
        dist_to_bb = distance(vehicle_state["pos"], qrbox_pos)
        percent_of_img = get_percent_of_image(qr_corners, image)

        cv2.imshow("car view", np.array(image)[:, :, ::-1])
        cv2.waitKey(1)

        with torch.no_grad():
            prediction = model(model.process_image(image).to(device))

        if bbox_img is not None and kph > 29 and dist_to_bb < cuton:
            detected_percents.append(percent_of_img)
            detected_distances.append(dist_to_bb)
            detected_runtimes.append(runtime)
            sequence.append(
                {"image": model.process_image(image)[0], "bbox": qr_corners}
            )
            unperturbed_seq.append(
                {"image": model.process_image(image)[0], "bbox": qr_corners}
            )
            ys.append(prediction)
            imagecount += 1

        if kph > 29 and not is_billboard_fully_viewable(image, qr_corners):
            print("Billboard no longer viewable")
            outcome = "R2NT"
            break

        # control params
        dt = (sensors["timer"]["time"] - start_time) - runtime
        steering = float(prediction[0][0])
        if abs(steering) > 0.125 and kph > 30:
            setpoint = 30
        else:
            setpoint = overall_throttle_setpoint
        throttle = throttle_PID(kph, dt)
        sim.drive(throttle=throttle, steering=steering, brake=0.0)

        # collect metrics
        steering_inputs.append(steering)
        percents.append(percent_of_img)
        runtimes.append(runtime)
        distances.append(dist_to_bb)
        traj.append(vehicle_state["front"])
        kphs.append(kph)
        final_img = image
        all_ys.append(steering)

        rs = calc_points_of_reachable_set(vehicle_state)
        road_seg = sim.nearest_seg()  # sim.roadmiddle, vehicle_state['front'])
        x = intersection_of_RS_and_road(rs, road_seg)
        # plot_intersection_with_CV2(vehicle.state, rs, road_seg, x, vehicle.get_bbox())

        if damage > 0.0:
            print(f"Damage={damage:.3f}, exiting...")
            outcome = "D={}".format(round(damage, 2))
            break
        if sim.has_car_left_track():
            print("Left track, exiting...")
            outcome = "LT"
            break

        sim.bng.step(1, wait=True)
        # last_steering_from_sim = sensors['electrics']['steering_input']
    unperturbed_steer = steering_inputs
    cv2.destroyAllWindows()

    print(f"Sequence collected; {len(unperturbed_seq)=}")
    print(f"dist_to_bb_cutoff={dist_to_bb}")
    deviation, dists, avg_dist = sim.calc_deviation_from_center(traj)
    results = {
        "runtime": round(runtime, 3),
        "damage": damage,
        "kphs": kphs,
        "traj": traj,
        "final_img": final_img,
        "deviation": deviation,
        "dists": dists,
        "avg_dist": avg_dist,
        "ys": ys,
        "outcome": outcome,
        "all_ys": all_ys,
        "dist_to_bb": dist_to_bb,
        "unperturbed_seq": unperturbed_seq,
    }
    return sequence, results


def run_scenario_with_perturbed_billboard(sim, model, pert_billboard, dist_to_bb_cuton=None, dist_to_bb_cutoff=None):
    global setpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim.restart()
    sensors = sim.get_sensor_readings()
    integral, runtime = 0.0, 0.0
    prev_error = setpoint
    kphs, traj = [], []
    damage = sensors["damage"]["damage"]
    start_time = sensors["timer"]["time"]
    final_img = None
    perturbed_predictions, unperturbed_predictions = [], []
    sequence, steering_vector = [], []
    pos_window = np.zeros((10, 3))
    billboard_viewable = True
    outcomestring = ""
    all_ys = []
    runtimes, detected_runtimes = [], []
    percents, detected_percents = [], []
    distances, detected_distances = [], []
    angleerror_distances, angleerror_runtimes = [], []
    qrbox_pos = list(sim.qr_positions[0][0])
    while damage <= 0:
        sensors = sim.get_sensor_readings()
        vehicle_state = sim.get_vehicle_state()
        damage = sensors["damage"]["damage"]
        kph = ms_to_kph(sensors["electrics"]["wheelspeed"])
        runtime = sensors["timer"]["time"] - start_time
        origimage = sensors["front_cam"]["colour"].convert("RGB")
        image = sensors["front_cam"]["colour"].convert("RGB")
        colorseg_img = sensors["front_cam"]["annotation"].convert("RGB")

        dist_to_bb = distance(vehicle_state["pos"], qrbox_pos)
        qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
        percent_of_img = get_percent_of_image(qr_corners, image)
        runtimes.append(runtime)
        percents.append(percent_of_img)
        distances.append(dist_to_bb)

        rs = calc_points_of_reachable_set(vehicle_state)
        road_seg = sim.nearest_seg()
        x = intersection_of_RS_and_road(rs, road_seg)
        # plot_intersection_with_CV2(vehicle.state, rs, road_seg, round(x, 2), vehicle.get_bbox(), traj=traj)

        if bbox_img is not None and kph > 29:
            detected_runtimes.append(runtime)
            detected_percents.append(percent_of_img)
            detected_distances.append(dist_to_bb)
            sequence.append({"image": image, "bbox": qr_corners})
            steering_vector.append(unpert_prediction)
            image_pert = overlay_transparent(
                np.array(origimage), pert_billboard, np.asarray(qr_corners[0])
            )
            billboard_viewable = is_billboard_fully_viewable(origimage, qr_corners)

        with torch.no_grad():
            model = model.to(device)
            origimg = model.process_image(origimage).to(device)
            unpert_prediction = float(model(origimg).cpu()[0][0])
            origimg = origimg.to(torch.device("cpu"))
            if bbox_img is not None and kph > 29:
                deviceimg_pert = model.process_image(image_pert).to(device)
                prediction_pert = float(model(deviceimg_pert).cpu()[0][0])
                steering = prediction_pert
            else:
                steering = unpert_prediction

        cv2.imshow("car view", np.array(image)[:, :, ::-1])
        cv2.waitKey(1)

        # control params
        dt = (sensors["timer"]["time"] - start_time) - runtime
        all_ys.append(unpert_prediction)
        if abs(unpert_prediction) > 0.2:
            setpoint = 30
        else:
            setpoint = 40
        throttle = throttle_PID(kph, dt)
        sim.drive(throttle=throttle, steering=steering, brake=0.0)

        if bbox_img is not None and kph > 29:
            # origimg = model.process_image(origimage).to(device)
            # unpert_prediction = float(model(origimg).cpu()[0][0])
            unperturbed_predictions.append(unpert_prediction)
            perturbed_predictions.append(prediction_pert)
            angleerror_runtimes.append(runtime)
            angleerror_distances.append(dist_to_bb)
        traj.append(vehicle_state["front"])
        kphs.append(kph)
        final_img = image
        pos_window = np.roll(pos_window, 3)
        pos_window[0] = vehicle_state["pos"]
        # stopping conditions
        spawn = sim.spawn_point(default_scenario, default_spawnpoint)
        if damage > 0.0:
            outcomestring = f"D={damage:2.1f}"
            print(f"Damage={damage:.3f} at timestep={runtime:.2f}, exiting...")
            break
        elif sim.has_car_left_track():
            outcomestring = f"LT"
            print("Left track, exiting...")
            break
        elif not billboard_viewable and sim.returned_to_expected_traj(pos_window):
            outcomestring = "R2NT"
            print("Returned to normal trajectory, exiting...")
            break
        elif distance(spawn["pos"], vehicle_state["pos"]) > 65 and runtime > 10:
            outcomestring = "2FAR"
            print("Too far from sequence, exiting...")
            break
        sim.bng.step(1, wait=True)

    mse = mean_squared_error(unperturbed_predictions, perturbed_predictions)
    error = np.array(unperturbed_predictions) - np.array(perturbed_predictions)
    deviation, dists, avg_dist = sim.calc_deviation_from_center(traj)
    results = {
        "runtime": round(runtime, 3),
        "damage": damage,
        "kphs": kphs,
        "traj": traj,
        "final_img": final_img,
        "deviation": deviation,
        "mse": mse,
        "dists": dists,
        "avg_dist": avg_dist,
        "error": error,
        "perturbed_predictions": perturbed_predictions,
        "outcome": outcomestring,
        "all_ys": all_ys,
    }
    return results


def distance2D(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def law_of_cosines(A, B, C):
    dist_AB = distance2D(A[:2], B[:2])
    dist_BC = distance2D(B[:2], C[:2])
    dist_AC = distance2D(A[:2], C[:2])
    return math.acos(
        (math.pow(dist_AB, 2) + math.pow(dist_AC, 2) - math.pow(dist_BC, 2)) / (2 * dist_AB * dist_AC)
    )


def car_facing_billboard(vehicle_state):
    global qr_positions
    center_billboard = qr_positions[0][0]
    alpha = law_of_cosines(vehicle_state["front"], vehicle_state["pos"], center_billboard)
    return math.degrees(alpha) > 179.0


def run_scenario_for_superdeepbillboard(
    sim, model, direction,
    dist_to_bb_cuton=26, dist_to_bb_cutoff=26,
    bb_size=5, iterations=100, noise_level=25,
    input_divers=True,
    loss_fxn="inv23",
    device=torch.device("cuda"),
):
    global integral, prev_error, setpoint
    integral, runtime = 0.0, 0.0
    prev_error = setpoint
    model = model.to(device)
    sim.restart()
    sensors = sim.get_sensor_readings()

    kphs, traj = [], []
    damage = sensors["damage"]["damage"]
    final_img, outcome = None, None
    perturbed_predictions, unperturbed_predictions = [], []
    start_time = sensors["timer"]["time"]
    sequence, steering_vector, pert_billboards = [], [], []
    ys, all_ys = [], []
    bb_viewed_window = np.ones((10))
    detected_runtimes, runtimes, angleerror_runtimes = [], [], []
    detected_percents, percents = [], []
    detected_distances, distances = [], []
    qrbox_pos = list(sim.qr_positions[0][0])
    MAE = 0
    while damage <= 0:
        sensors = sim.get_sensor_readings()
        vehicle_state = sim.get_vehicle_state()
        last_steering_from_sim = sensors["electrics"]["steering_input"]
        dist_to_bb = distance(vehicle_state["pos"], qrbox_pos)
        kph = ms_to_kph(sensors["electrics"]["wheelspeed"])
        damage = sensors["damage"]["damage"]
        runtime = sensors["timer"]["time"] - start_time
        origimage = sensors["front_cam"]["colour"].convert("RGB")
        image = sensors["front_cam"]["colour"].convert("RGB")
        colorseg_img = sensors["front_cam"]["annotation"].convert("RGB")

        qr_corners, bbox_img = get_qr_corners_from_colorseg_image(colorseg_img)
        runtimes.append(runtime)
        percent_of_img = get_percent_of_image(qr_corners, image)
        percents.append(percent_of_img)
        distances.append(dist_to_bb)
        # stopping conditions
        if bbox_img is None and kph > 29 and round(dist_to_bb, 0) <= dist_to_bb_cuton:
            bb_viewed_window = np.roll(bb_viewed_window, 1)
            bb_viewed_window[0] = 0
            if sum(bb_viewed_window) < len(bb_viewed_window) / 2:
                print("Billboard window no longer viewable")
                outcome = "BBNV"
                break
        if (bbox_img is not None and kph > 29 and round(dist_to_bb, 0) <= dist_to_bb_cuton):
            detected_runtimes.append(runtime)
            detected_distances.append(dist_to_bb)
            detected_percents.append(percent_of_img)
            rs = calc_points_of_reachable_set(vehicle_state)
            road_seg = sim.nearest_seg()  # sim.roadmiddle, vehicle.state['front'])
            x = round(intersection_of_RS_and_road(rs, road_seg), 2)
            # plot_intersection_with_CV2(vehicle.state, rs, road_seg, x, vehicle.get_bbox())

            # stopping conditions
            if damage > 0.0:
                print(
                    "Damage={} at timestep={}, exiting...".format(
                        damage, round(runtime, 2)
                    )
                )
                outcome = "D={}".format(round(damage, 2))
                break

            if (kph > 29 and abs(x - dist_to_bb_cutoff) <= 0.02) or (
                kph > 29 and x <= dist_to_bb_cutoff
            ):
                print(f"RS overlap={x}, exiting...")
                outcome = f"RS overlap={x}"
                break

        # Run SuperDeepbillboard
        if (
            bbox_img is not None
            and kph > 29
            and round(dist_to_bb, 0) <= dist_to_bb_cuton
        ):  # and (steps % 2 < 1):
            sequence.append(
                {
                    "image": model.process_image(image)[0],
                    "bbox": qr_corners,
                    "colorseg_img": model.process_image(colorseg_img)[0],
                }
            )
            # image.save(f"{new_sampletaking_dir}/sample-{len(sequence)}.jpg")
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

            pert_billboard, y, MAE = superdeepbillboard(
                model,
                sequence2,
                direction,
                copy.deepcopy(steering_vector2),
                device=device,
                bb_size=bb_size,
                iterations=iterations,
                noise_level=noise_level,
                dist_to_bb=dist_to_bb,
                input_divers=input_divers,
                loss_fxn=loss_fxn,
            )
            steering_vector.append(last_steering_from_sim)
            model = model.to(device)
            ys = y
            pert_billboards.append(copy.deepcopy(pert_billboard))
            image = overlay_transparent(
                np.array(origimage), pert_billboard, np.asarray(qr_corners[0])
            )

        rs = calc_points_of_reachable_set(vehicle_state)
        road_seg = sim.nearest_seg()
        x = intersection_of_RS_and_road(rs, road_seg)
        x = round(x, 2)
        print(f"{dist_to_bb=:.1f}\tRS overlap={x}")
        plot_intersection_with_CV2(vehicle_state, rs, road_seg, x, sim.vehicle.get_bbox())

        cv2.imshow("car view", np.array(image)[:, :, ::-1])
        cv2.waitKey(1)

        with torch.no_grad():
            deviceimg = model.process_image(np.asarray(image)).to(device)
            prediction = model(deviceimg)
            prediction = prediction.cpu()

        # control params
        dt = (sensors["timer"]["time"] - start_time) - runtime
        steering = float(prediction[0][0])
        if abs(steering) > 0.2:
            setpoint = 30
        else:
            setpoint = 40
        throttle = throttle_PID(kph, dt)
        sim.drive(throttle=throttle, steering=steering, brake=0.0)
        all_ys.append(steering)

        if (
            bbox_img is not None
            and kph > 29
            and round(dist_to_bb, 0) <= dist_to_bb_cuton
        ):
            origimg = model.process_image(np.asarray(origimage)).to(device)
            unperturbed_predictions.append(float(model(origimg).cpu()[0][0]))
            perturbed_predictions.append(steering)
            angleerror_runtimes.append(runtime)
            traj.append(vehicle_state["front"])
            kphs.append(kph)
        final_img = image
        sim.bng.step(1, wait=True)
    cv2.destroyAllWindows()
    mse = mean_squared_error(unperturbed_predictions, perturbed_predictions)
    deviation, dists, avg_dist = sim.calc_deviation_from_center(traj, sim.centerline)
    results = {
        "runtime": round(runtime, 3),
        "damage": damage,
        "kphs": kphs,
        "traj": traj,
        "final_img": final_img,
        "deviation": deviation,
        "mse": mse,
        "steering_vector": steering_vector,
        "avg_dist": avg_dist,
        "ys": ys,
        "outcome": outcome,
        "all_ys": all_ys,
        "num_billboards": len(pert_billboards),
        "MAE": MAE,
    }
    print(f"number of billboards: {len(pert_billboards)}")
    return pert_billboards, results


def make_results_dirs(newdir, technique, bbsize, nl, its, cuton, rsc, input_div=None, timestr=None):
    new_results_dir = "results/{}/results-{}-{}-{}-{}-cuton{}-rs{}-inputdiv{}-{}-{}".format(
                        newdir, technique, bbsize, nl, its, cuton, rsc, input_div, timestr,
                        "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)),
    )
    if not os.path.isdir(new_results_dir):
        os.mkdir(new_results_dir)
    training_file = f"{new_results_dir}/results.pickle"
    return new_results_dir, training_file


def main():
    global new_results_dir, newdir, default_scenario, default_spawnpoint, unperturbed_traj
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    direction = "left"
    techniques = ["sdbb"]  # [["sdbb", "dbb-orig", ] ,
    model_name = "DAVE2v3.pt"
    model = torch.load(f"../models/weights/{model_name}", map_location=device).eval()
    lossname, new_results_dir = "", ""
    bbsizes = [5, 10, 15]
    iterations = [50]
    noiselevels = [15]
    rscs = [0.60]
    cutons = [21]
    input_divs = [False]
    sim = Simulator(
        scenario_name=default_scenario,
        spawnpoint_name=default_spawnpoint,
        path2sim=args.path2src,
        steps_per_sec=15,
    )
    samples = 3
    newdir = "Experiment-{}-{}-{}".format(
        default_scenario, default_spawnpoint,
        "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)),
    )
    if not os.path.isdir("results/{}".format(newdir)):
        os.mkdir("results/{}".format(newdir))

    for cuton in cutons:
        sequence, unperturbed_results = run_scenario_to_collect_sequence(sim, model, cuton, device=device)
        unperturbed_traj = unperturbed_results["traj"]
        for rsc in rscs:
            for bbsize in bbsizes:
                for its in iterations:
                    for nl in noiselevels:
                        for technique in techniques:
                            for input_div in input_divs:
                                for i in range(samples):
                                    all_trajs, all_outcomes = [], []
                                    localtime = time.localtime()
                                    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,localtime.tm_min)

                                    if technique == "dbb":
                                        lossname = "MDirE"
                                        new_results_dir, training_file = make_results_dirs(
                                            newdir, technique, bbsize, nl, its, cuton, rsc, timestr
                                        )
                                        results = run_scenario(
                                            sim, model, direction,
                                            device=device, collect_sequence_results=unperturbed_results,
                                            bb_size=bbsize, iterations=its, noise_level=nl,
                                            dist_to_bb_cuton=cuton, resultsdir=new_results_dir,
                                        )
                                    elif technique == "dbb-orig":
                                        lossname = "MDirE"
                                        new_results_dir, training_file = make_results_dirs(
                                            newdir, technique, bbsize,
                                            1000, its, 28, rsc,
                                            input_div=False, timestr=timestr,
                                        )
                                        results = run_scenario(
                                            sim, model, direction,
                                            device=device, collect_sequence_results=unperturbed_results,
                                            bb_size=bbsize, iterations=its, noise_level=1000, dist_to_bb_cuton=28,
                                            resultsdir=new_results_dir, input_divers=False,
                                        )
                                    elif technique == "dbb-plus":
                                        lossname = "MDirE"
                                        new_results_dir, training_file = make_results_dirs(
                                            newdir, technique, bbsize, nl, its, cuton, rsc, input_div, timestr
                                        )
                                        results = run_scenario(
                                            sim, model, direction,
                                            device=device, collect_sequence_results=unperturbed_results,
                                            bb_size=bbsize, iterations=its, noise_level=nl,
                                            dist_to_bb_cuton=cuton, input_divers=input_div, resultsdir=new_results_dir,
                                        )
                                    else:
                                        lossname = "inv23"  # 'MDirE'
                                        new_results_dir, training_file = make_results_dirs(
                                            newdir, technique, bbsize, nl, its, cuton, rsc, input_div, timestr
                                        )
                                        results = run_scenario_superdeepbillboard(
                                            sim, model, direction,
                                            device=device, collect_sequence_results=unperturbed_results,
                                            bb_size=bbsize, iterations=its, noise_level=nl,
                                            dist_to_bb_cuton=cuton, dist_to_bb_cutoff=rsc, resultsdir=new_results_dir,
                                            input_divers=input_div,  loss_fxn=lossname,
                                        )
                                        all_trajs.append(results["pertrun_traj"])
                                        all_outcomes.append(results["pertrun_outcome"])
                                    cv2.destroyAllWindows()
                                    all_trajs.extend(results["testruns_trajs"])
                                    all_outcomes.extend(results["testruns_outcomes"])

                                    write_results(training_file, results, all_trajs, unperturbed_traj, model._get_name(),
                                                    technique, direction, lossname, bbsize, its, nl
                                    )
                                    plot_deviation(all_trajs, unperturbed_traj,
                                                   "{}-{}-{}\nDOF{}-noisevar{}-cuton{}".format(technique, direction, lossname, bbsize, nl, cuton),
                                                    sim.centerline_interpolated, sim.roadleft, sim.roadright, all_outcomes, resultsdir=new_results_dir,
                                    )
    sim.bng.close()
    print(f"Finished in {time.time() - start} seconds")


if __name__ == "__main__":
    logging.getLogger("matplotlib.font_manager").disabled = True
    warnings.filterwarnings("ignore")
    logging.getLogger("PIL").setLevel(logging.INFO)
    main()
