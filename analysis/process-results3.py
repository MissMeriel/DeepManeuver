import numpy as np
from matplotlib import pyplot as plt
import logging, random, string, time, os
from tabulate import tabulate
from pathlib import Path
import argparse, sys
import statistics, math
from ast import literal_eval
from PIL import Image
import pickle
import itertools
from tqdm import tqdm
import pandas as pd
import re
import itertools
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--dataset', type=str, default="../data", help='parent directory of results dataset')
    parser.add_argument('-f', '--figure', type=str, help='table or figure to generate (choose from: table1 table2 table3 table5 table6 figure5 figure6 figure7 figure8)')
    args = parser.parse_args()
    return args

def plot_deviation(trajectories, unperturbed_traj, collection_run, model, centerline, left, right, qr_positions, outcomes, xlim=[245, 335], ylim=[-123, -20], savefile="paperfigures/image-"):
    x, y = [], []
    for point in unperturbed_traj:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, label="Unperturbed", linewidth=10)
    x, y = [], []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-", label="road")
    x, y = [], []
    for point in left:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-", label="road")
    x, y = [], []
    for point in right:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-", label="road")
    if collection_run is not None:
        x = [point[0] for point in collection_run]
        y = [point[1] for point in collection_run]
        plt.plot(x, y, color='tab:orange', linewidth=7)
    colorcycle = ['tab:blue','tab:brown','tab:pink','tab:olive','fuchsia','mediumseagreen','indigo','tab:purple','tab:green','darkblue','tab:gray']
    for i, t in enumerate(trajectories):
        x = [point[0] for point in t]
        y = [point[1] for point in t]
        plt.plot(x, y, color=colorcycle[i], linewidth=2.5, alpha=0.75)
        i += 1
    x,y = [], []
    for i, position in enumerate(qr_positions):
        if i < 3:
            x.append(position[0][0]-1)
        else:
            x.append(position[0][0])
        y.append(position[0][1])
    plt.plot(x, y, 'r-', linewidth=5)
    plt.title('Trajectories with {}'.format(model), fontsize=5)
    plt.xlim(xlim)
    plt.ylim(ylim)
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    plt.savefig("{}-{}.jpg".format(savefile, model.replace("\n", "-"), randstr))
    plt.show()
    plt.pause(0.1)
    # ax = plt.gca()
    # ax.relim()

def plot_figure2(trajectories, collection_run, model, centerline, left, right, qr_positions, outcomes, xlim=[245, 335], ylim=[-123, -20], savefile="paperfigures/image-"):
    x, y = [], []
    for point in centerline:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-")
    x, y = [], []
    for point in left:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-")
    x, y = [], []
    for point in right:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, "k-")
    if collection_run is not None:
        x = [point[0] for point in collection_run]
        y = [point[1] for point in collection_run]
        plt.plot(x, y, color='tab:blue', linewidth=5, label="Unpert.trajectory")
    for i, t in enumerate(trajectories):
        x = [point[0] for point in t]
        y = [point[1] for point in t]
        plt.plot(x, y, color="tab:orange", linewidth=2.5, alpha=0.75, label="DeepManeuver")
        i += 1
    x,y = [], []
    for i, position in enumerate(qr_positions):
        if i < 3:
            x.append(position[0][0]-1)
        else:
            x.append(position[0][0])
        y.append(position[0][1])
    ax = plt.gca()

    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.plot(x, y, 'r-', linewidth=5)
    plt.plot([245, 254.5], [-24, -29], "bo", linewidth=10, label="Target waypoints")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize=14)
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    plt.savefig("{}-{}.jpg".format(savefile, model.replace("\n", "-"), randstr))
    plt.show()
    plt.pause(0.1)

def plot_errors(errors, filename="images/errors.png"):
    plt.title("Errors")
    for ei, e in enumerate(errors):
        plt.plot(range(len(e)), e, label=f"Error {ei}")
    plt.savefig(filename)
    plt.show()
    plt.pause(0.1)

    plt.title("Error Distributions per Run")
    avgs = []
    for ei, e in enumerate(errors):
        plt.scatter(np.ones((len(e)))*ei, e, s=5) #, label=f"Error {ei}")
        avgs.append(float(sum(e)) / len(e))
    plt.plot(range(len(avgs)), avgs)
    plt.savefig(filename.replace(".png", "-distribution.png"))
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

def unpickle_results(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results

def lineseg_dists(p, a, b):
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)
    h = np.maximum.reduce([s, t, np.zeros(len(s))])
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    return np.hypot(h, c)

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
    # global expected_trajectory
    expected_trajectory = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            # print(line)
            line = literal_eval(line)
            expected_trajectory.append(line)
    return expected_trajectory


def get_outcomes(results):
    outcomes_percents = {"B":0, "D":0, "LT":0, "R2NT":0, "2FAR":0, 'GOAL':0}
    for outcome in results['testruns_outcomes']:
        if "BULLSEYE-D" in outcome:
            outcomes_percents["B"] += 1
        elif "GOAL" in outcome:
            outcomes_percents["GOAL"] += 1
        elif "D=" in outcome:
            outcomes_percents["D"] += 1
        elif "R2NT" in outcome:
            outcomes_percents["R2NT"] +=1
        elif "LT" in outcome:
            outcomes_percents["LT"] += 1
        elif "2FAR" in outcome:
            outcomes_percents["2FAR"] += 1
    return outcomes_percents

def get_metrics(results, outcomes):
    # get metrics on perturbation run
    centerline = intake_lap_file("../tools/simulation/posefiles/DAVE2v3-lap-trajectory.txt")
    deviation, dists, avg_dist = calc_deviation_from_center(centerline, results['unperturbed_traj'])
    if results['technique'] == 'sdbb':
        outstring = f"RESULTS FOR {results['technique']} {results['modelname']} bb={results['bbsize']} noisedenom={results['nl']} iters={results['its']} direction={results['direction']}: \n" \
                    f"Avg. deviation from expected trajectory: \n" \
                    f"unperturbed:\t{results['unperturbed_deviation']}\n" \
                    f"pert.run:\t{results['pertrun_deviation']}\n" \
                    f"test:\t\t{sum(results['testruns_deviation']) / float(len(results['testruns_deviation']))} \n" \
                    f"Avg. distance from expected trajectory:\n" \
                    f"unperturbed:\t{sum(results['unperturbed_dists']) / float(len(results['unperturbed_dists']))}\n" \
                    f"pert.run:\t\t{results['pertrun_dist']}\n" \
                    f"perturbed:  \t{sum(results['testruns_dists']) / float(len(results['testruns_dists']))}\n" \
                    f"Pred. angle error measures in test runs:\n" \
                    f"mse:      \t\t{sum(results['testruns_mse']) / float(len(results['testruns_mse']))}\n" \
                    f"avg error:\t{sum(results['testruns_errors']) / float(len(results['testruns_errors']))}\n" \
                    f"time_to_run_technique:\t{results['time_to_run_technique']}\n" \
                    f"num_billboards:\t\t{results['num_billboards']}\n"
                    # f"testruns outcomes: \t{results['testruns_outcomes']}"
    else:
        outstring = f"RESULTS FOR {results['technique']} {results['modelname']} bb={results['bbsize']} noisedenom={results['nl']} iters={results['its']} direction={results['direction']}: \n" \
                    f"Avg. deviation from expected trajectory: \n" \
                    f"unperturbed:\t{results['unperturbed_deviation']}\n" \
                    f"test:\t\t\t{sum(results['testruns_deviation']) / float(len(results['testruns_deviation']))} \n" \
                    f"Avg. distance from expected trajectory:\n" \
                    f"unperturbed:\t{sum(results['unperturbed_dists']) / float(len(results['unperturbed_dists']))}\n" \
                    f"perturbed:\t\t{sum(results['testruns_dists']) / float(len(results['testruns_dists']))}\n" \
                    f"Pred. angle error measures in test runs:\n" \
                    f"mse:\t\t{sum(results['testruns_mse']) / float(len(results['testruns_mse']))}\n" \
                    f"avg error:\t\t{sum(results['testruns_errors']) / float(len(results['testruns_errors']))}\n" \
                    f"time_to_run_technique:\t{results['time_to_run_technique']}\n"
                    # f"num_billboards:\t\t{results['num_billboards']}"
                    # f"testruns outcomes: \t{results['testruns_outcomes']}"
    return outstring

def load_trackdef():
    dir = "../tools/simulation/posefiles/beamng-industrial-racetrack"
    with open(f"{dir}/actual-middle.pickle", "rb") as f:
        middle = pickle.load(f)
    with open(f"{dir}/road-left.pickle", "rb") as f:
        roadleft = pickle.load(f)
    roadright = pickle.load(open(f"{dir}/road-right.pickle", "rb"))
    return middle, roadleft, roadright

def parse_qr_positions_file(filename='posefiles/qr_box_locations.txt'):
    qr_positions = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(' ')
            pos = line[0].split(',')
            pos = tuple([float(i) for i in pos])
            rot_quat = line[1].split(',')
            rot_quat = tuple([float(j) for j in rot_quat])
            qr_positions.append([pos, rot_quat])
    return qr_positions

def get_xy_lim(topo):
    if topo == "topo1":
        # return [265, 300], [-60, -30]
        return [260, 295], [-60, -30]
    elif topo == "topo11":
        return [275, 305], [-90, -30]
    elif topo == "topo2" or topo == "topo21":
        # return [180, 120], [-120, -180]
        return [180, 125], [-125, -180]
    elif topo == "topo3":
        return [150, 210], [-160, -100]
    elif topo == "topo31":
        return [180, 240], [-190, -110]
    elif topo == "cutcorner": # cut corner
        return [225, 270], [-60, -10]
    elif topo == "lanechange": # lane change
        return [180, 225], [-200, -150]
    elif topo == "bullseye": # bullseye
        return [160, 240], [-120, -40]

def get_topo(d, pattern):
    try:
        topo = re.search(pattern, d)[0]
    except:
        if "cutcorner" in d:
            topo = "cutcorner"
        elif "bullseye" in d:
            topo = "bullseye"
        elif "lanechange" in d:
            topo = "lanechange"
    return topo

def generate_figure2(dir, outdir="./"):
    middle, roadleft, roadright = load_trackdef()
    dirs = [_ for _ in os.listdir(dir) if os.path.isdir("/".join([dir, _]))]
    expected_trajectory = intake_lap_file("../tools/simulation/posefiles/DAVE2v1-lap-trajectory.txt")
    for d in dirs:
        qr_positions = parse_qr_positions_file(f'../tools/simulation/posefiles/qr_box_locations-cutcorner.txt')
        results = unpickle_results(f"{dir}{d}/results.pickle")
        xlim, ylim = [220, 280], [-60, -10]
        plot_figure_2(results["testruns_trajs"][:1], expected_trajectory, d, middle, roadleft, roadright, qr_positions, get_outcomes(results),
                       xlim=xlim, ylim=ylim, savefile=f"{outdir}{d}")

def generate_figures_568(resultsDir, outdir="./"):
    middle, roadleft, roadright = load_trackdef()
    pattern = re.compile("topo[0-9]+")
    dirs = [_ for _ in os.listdir(resultsDir) if os.path.isdir("/".join([resultsDir, _]))]
    for d in dirs:
        topo = get_topo(d, pattern)
        qr_positions = parse_qr_positions_file(f'../tools/simulation/posefiles/qr_box_locations-{topo}.txt')
        results = unpickle_results(f"{resultsDir}/{d}/results.pickle")
        xlim, ylim = get_xy_lim(topo)
        try:
            collection_run = results["pertrun_traj"]
        except:
            collection_run = None
        plot_deviation(results["testruns_trajs"], results["unperturbed_traj"], collection_run, d, middle, roadleft, roadright, qr_positions, get_outcomes(results),
                       xlim=xlim, ylim=ylim, savefile=f"{outdir}{d}")

def generate_AAE_figure(resultsDir):
    fileExt = r".pickle"
    aaes = []
    dirs = [f"{resultsDir}/{_}" for _ in os.listdir(f"{resultsDir}") if os.path.isdir("/".join([resultsDir, _]))]
    # key = "results-sdbb-10-15-400-cuton28-rs0.6-inputdivFalse-"
    for d in dirs:
        # if key in d:
        results_files = [_ for _ in os.listdir(d) if _.endswith(fileExt)]
        for file in results_files:
            results = unpickle_results("/".join([d, file]))
            aaes.extend(results["testruns_errors"])

    def adjustFigAspect(fig, aspect=1):
        xsize, ysize = fig.get_size_inches()
        minsize = min(xsize, ysize)
        xlim = .4 * minsize / xsize
        ylim = .4 * minsize / ysize
        if aspect < 1:
            xlim *= aspect
        else:
            ylim /= aspect
        fig.subplots_adjust(left=.5 - xlim, right=.5 + xlim, bottom=.5 - ylim, top=.5 + ylim)

    fig = plt.figure()
    adjustFigAspect(fig, aspect=2.5)
    ax = fig.add_subplot(111)

    # ax.boxplot(aaes, vert=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])
    ax.set_xlabel('Steering angle error', fontsize=12)
    ax.set_aspect(2)
    ax.set_aspect('auto')

    df = pd.DataFrame(dict(AAE=aaes))
    _, bp = pd.DataFrame.boxplot(df, return_type='both')
    outliers = [flier.get_ydata() for flier in bp["fliers"]]
    boxes = [box.get_ydata() for box in bp["boxes"]]
    medians = [median.get_ydata() for median in bp["medians"]]
    whiskers = [whiskers.get_ydata() for whiskers in bp["whiskers"]]
    print(df.shape)
    print("Outliers: ", outliers, len(outliers[0]))
    print(sum([i < 0 for i in outliers[0]]))
    print("Boxes: ", boxes)
    print("Medians: ", medians)
    print("Whiskers: ", whiskers)

    # plt.boxplot(aaes, vert=False)
    plt.show()
    plt.pause(0.01)

def tablegen(resultsDir):
    fileExt = r".pickle"
    allDirs = [f"{resultsDir}/{_}" for _ in os.listdir(f"{resultsDir}") if os.path.isdir("/".join([resultsDir, _]))]
    results_map = {}
    count = 0
    for d in tqdm(allDirs):
        bullseyes, crashes, goals = [], [], []
        aes, dists = [], []
        num_bbs, MAE_coll, ttrs = [], [], []

        tdirs = [f"{d}/{_}" for _ in os.listdir(d) if os.path.isdir(f"{d}/{_}")]
        for td in tdirs:
            count += 1
            results_files = [_ for _ in os.listdir(td) if _.endswith(fileExt)]
            for file in results_files:
                results = unpickle_results("/".join([td, file]))
                outcomes_percents = get_outcomes(results)
                bullseyes.append(outcomes_percents['B'])
                crashes.append(outcomes_percents['D'] + outcomes_percents['LT'] + outcomes_percents['B'])
                goals.append(outcomes_percents['GOAL'])
                aes.extend(results['testruns_errors'])
                dists.extend(results['testruns_dists'])
                ttrs.append(results['time_to_run_technique'])
                num_bbs.append(results['num_billboards'])
                # MAE_coll.append(results['MAE_collection_sequence'])
        td_path = Path(td)
        results_key = f"{td_path.parts[-2]}"
        results_map[results_key] = {"bullseyes":bullseyes, "crashes":crashes, "goals":goals,
                            "aes":aes, "dists":dists, "ttrs":ttrs, "num_bbs":num_bbs, "MAE_coll":MAE_coll}

    listified_map = [[key, f"{sum(results_map[key]['crashes'])/ (10*len(results_map[key]['crashes'])):.3f}",
                      f"{round(sum(results_map[key]['dists']) / len(results_map[key]['dists']), 4)}",
                      f"{round(sum(results_map[key]['aes']) / len(results_map[key]['aes']), 3)}",
                      # f"{sum(results_map[key]['MAE_coll']) / len(results_map[key]['MAE_coll']):.3f}",
                      len(results_map[key]['crashes'])] for key in results_map.keys()]
    print(tabulate(listified_map, headers=["technique", "crash rate", "dist from exp traj", "AAE", "samplecount"], tablefmt="github"))



def table1(resultsDir):
    fileExt = r".pickle"
    allDirs = [f"{resultsDir}/{_}" for _ in os.listdir(f"{resultsDir}") if os.path.isdir("/".join([resultsDir, _]))]
    results_map = {}
    for d in tqdm(allDirs):
        scenariodirs = ["/".join([d, dd]) for dd in os.listdir(d) if os.path.isdir("/".join([d, dd]))]
        for scd in scenariodirs:
            bullseyes, crashes, goals = [], [], []
            aes, dists = [], []
            num_bbs, MAE_coll, ttrs = [], [], []
            tdirs = [f"{scd}/{_}" for _ in os.listdir(scd) if os.path.isdir(f"{scd}/{_}")]
            for td in tdirs:
                results_files = [f"{td}/{_}" for _ in os.listdir(td) if _.endswith(fileExt)]
                for file in results_files:
                    results = unpickle_results(file)
                    outcomes_percents = get_outcomes(results)
                    bullseyes.append(outcomes_percents['B'])
                    crashes.append(outcomes_percents['D'] + outcomes_percents['LT'] + outcomes_percents['B'])
                    goals.append(outcomes_percents['GOAL'])
                    aes.extend(results['testruns_errors'])
                    dists.extend(results['testruns_dists'])
                    ttrs.append(results['time_to_run_technique'])
                    num_bbs.append(results['num_billboards'])
                    MAE_coll.append(results['MAE_collection_sequence'])
            scd_path = Path(scd)
            results_key = f"{scd_path.parts[-2]} {scd_path.parts[-1]}"
            results_map[results_key] = {"bullseyes":bullseyes, "crashes":crashes, "goals":goals,
                                "aes":aes, "dists":dists, "ttrs":ttrs, "num_bbs":num_bbs, "MAE_coll":MAE_coll}

    listified_map = [[key, f"{sum(results_map[key]['crashes'])/ (10*len(results_map[key]['crashes'])):.3f}",
                      f"{sum(results_map[key]['dists']) / len(results_map[key]['dists']):.2f}",
                      f"{sum(results_map[key]['aes']) / len(results_map[key]['aes']):.3f}",
                      f"{sum(results_map[key]['MAE_coll']) / len(results_map[key]['MAE_coll']):.3f}",
                      len(results_map[key]['crashes'])] for key in results_map.keys()]
    print(tabulate(listified_map, headers=["technique", "crash rate", "dist from exp traj", "AAE", "expected AAE", "samplecount"], tablefmt="github"))


''' with bonferroni correction '''
def mann_whitney_u_test(resultsDir, dir1="", dir2="", key="results-sdbb-5-15-400-cuton28-rs0.6-inputdivFalse"):
    count = 0
    fileExt = r".pickle"
    dir1_dirs = [f"{results_parentdir}/{dir1}/{_}" for _ in os.listdir(f"{results_parentdir}/{dir1}") if
            os.path.isdir("/".join([results_parentdir, dir1, _]))]
    for d in dir1_dirs:
        if key in d and count < 50:
            count += 1
            results_files = [_ for _ in os.listdir(d) if _.endswith(fileExt)]

            for file in results_files:
                results = unpickle_results("/".join([d, file]))
                dir1_outcomes_percents = get_outcomes(results)
                dir1_metrics = get_metrics(results, dir1_outcomes_percents)

    dir2_dirs = [f"{results_parentdir}/{dir2}/{_}" for _ in os.listdir(f"{results_parentdir}/{dir2}") if
            os.path.isdir("/".join([results_parentdir, dir2, _]))]
    for d in dir2_dirs:
        if key in d:
            count += 1
            results_files = [_ for _ in os.listdir(d) if _.endswith(fileExt)]

            for file in results_files:
                results = unpickle_results("/".join([d, file]))
                dir2_outcomes_percents = get_outcomes(results)
                dir2_metrics = get_metrics(results, dir2_outcomes_percents)

    sigtest = mannwhitneyu(dir1_outcomes_percents, dir2_outcomes_percents)
    print(sigtest)

def kruskalwallis(resultsDir):
    fileExt = r".pickle"
    key1_results = {"success rate": [], "AAE": [], "ADOT": []}
    key2_results = {"success rate": [], "AAE": [], "ADOT": []}
    key3_results = {"success rate": [], "AAE": [], "ADOT": []}

    dirs = [f"{resultsDir}/{_}" for _ in os.listdir(f"{resultsDir}") if os.path.isdir(f"{resultsDir}/{_}")]
    res = 10
    key1 = ""
    key3 = "results-sdbb-10-15-400-cuton20-rs0.6-inputdivFalse-"
    key2 = "results-dbb-plus-10-15-400-cuton20-rs0.6-inputdivFalse-"
    for d in dirs:
        if key1 in d:
            results_files = [_ for _ in os.listdir(d) if _.endswith(fileExt)]
            for file in results_files:
                results = unpickle_results("/".join([d, file]))
                outcomes_percents = get_outcomes(results)
                key1_results["success rate"].append((outcomes_percents['D'] + outcomes_percents['LT'])/10)
                key1_results["AAE"].append(sum(results['testruns_errors']) / len(results['testruns_errors']))
                key1_results["ADOT"].append(sum(results['testruns_dists']) / len(results['testruns_dists']))
        if key2 in d:
            results_files = [_ for _ in os.listdir(d) if _.endswith(fileExt)]
            for file in results_files:
                results = unpickle_results("/".join([d, file]))
                outcomes_percents = get_outcomes(results)
                key2_results["success rate"].append((outcomes_percents['D'] + outcomes_percents['LT'])/10)
                key2_results["AAE"].append(sum(results['testruns_errors']) / len(results['testruns_errors']))
                key2_results["ADOT"].append(sum(results['testruns_dists']) / len(results['testruns_dists']))
        elif key3 in d:
            results_files = [_ for _ in os.listdir(d) if _.endswith(fileExt)]
            for file in results_files:
                results = unpickle_results("/".join([d, file]))
                outcomes_percents = get_outcomes(results)
                key3_results["success rate"].append((outcomes_percents['D'] + outcomes_percents['LT'])/10)
                key3_results["AAE"].append(sum(results['testruns_errors']) / len(results['testruns_errors']))
                key3_results["ADOT"].append(sum(results['testruns_dists']) / len(results['testruns_dists']))
    # sr_sig = kruskal(key1_results["success rate"], key2_results["success rate"], key3_results["success rate"])
    # print("SR compare all", sr_sig)
    # sr_sig = kruskal(key1_results["success rate"], key2_results["success rate"])
    # print("SR key1 vs key2", sr_sig)
    sr_sig = kruskal(key2_results["success rate"], key3_results["success rate"])
    print("SR key2 vs key3", sr_sig)
    # sr_sig = kruskal(key1_results["success rate"], key3_results["success rate"])
    # print("SR key1 vs key3", sr_sig)

    # adot_sig = kruskal(key1_results["ADOT"], key2_results["ADOT"], key3_results["ADOT"])
    # print("ADOT compare all", adot_sig)
    # sr_sig = kruskal(key1_results["ADOT"], key2_results["ADOT"])
    # print("ADOT key1 vs key2", sr_sig)
    sr_sig = kruskal(key2_results["ADOT"], key3_results["ADOT"])
    print("ADOT key2 vs key3", sr_sig)
    # sr_sig = kruskal(key1_results["ADOT"], key3_results["ADOT"])
    # print("ADOT key1 vs key3", sr_sig)

    # aae_sig = kruskal(key1_results["AAE"], key2_results["AAE"], key3_results["AAE"])
    # print("AAE compare all", aae_sig)
    # sr_sig = kruskal(key1_results["AAE"], key2_results["AAE"])
    # print("AAE key1 vs key2", sr_sig)
    sr_sig = kruskal(key2_results["AAE"], key3_results["AAE"])
    print("AAE key2 vs key3", sr_sig)
    # sr_sig = kruskal(key1_results["AAE"], key3_results["AAE"])
    # print("AAE key1 vs key3", sr_sig)


if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    args = parse_arguments()
    # table1 table2 table3 table5 table6 figure5 figure6 figure7 figure8
    if args.figure == "table1": # study table 1
        resultsDir = args.dataset + "/study/" + args.figure
        table1(resultsDir)
    elif args.figure == "table2": # study table 2
        resultsDir = args.dataset + "/study/" + args.figure
        tablegen(resultsDir)
    elif args.figure == "table3": # study table 3
        resultsDir = args.dataset + "/study/" + args.figure
        tablegen(resultsDir)
    elif args.figure == "table5": # appendix table 5
        resultsDir = args.dataset + "/appendix/" + args.figure
        tablegen(resultsDir)
    elif args.figure == "table6": # appendix table 6
        resultsDir = args.dataset + "/appendix/" + args.figure
        tablegen(resultsDir)
    elif args.figure == "figure2":
        resultsDir = args.dataset + "approach" + args.figure
        generate_figure2(resultsDir)
    elif args.figure == "figure5": # study figure 5
        resultsDir = args.dataset + "/study/" + args.figure
        generate_figures_568(resultsDir)
    elif args.figure == "figure6": # study figure 6
        resultsDir = args.dataset + "/study/" + args.figure
        generate_figures_568(resultsDir)
    elif args.figure == "figure7": # appendix figure 7
        resultsDir = args.dataset + "/appendix/" + args.figure
        generate_AAE_figure()
    elif args.figure == "figure8": # appendix figure 8
        resultsDir = args.dataset + "/appendix/" + args.figure
        generate_figures_568(resultsDir)
    else:
        print(f"\"{args.figure}\" not found. Please choose a valid figure id."
              f"\nOptions: table1 table2 table3 table5 table6 figure2 figure5 figure6 figure7 figure8")
    # kruskalwallis()

