import torch
import time
import random
import string
import os
import Demo1
from simulator import Simulator

import argparse



def main():

    PARSER = argparse.ArgumentParser(description="Process paths")
    PARSER.add_argument(
        "path2src", metavar="N", type=str, help="path to source parent dirs"
    )
    PARSER.add_argument("road_id", metavar="N", type=str, help="road identifier in BeamNG")
    ARGS = PARSER.parse_args()
    print(ARGS)

    bbsizes = [5, 10, 15]
    iterations = [400]
    noiselevels = [10, 15, 20, 1000]
    rscs = [0.60]
    cutons = [20, 24, 28]
    input_divs = [False, True]
    direction = ["left", "right"]
    techniques = ["deepman", "dbb-orig", "dbb"]
    samples = 50

    global new_results_dir, newdir, default_scenario, default_spawnpoint, unperturbed_traj
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    direction = "left"
    techniques = ["deepman", "dbb-orig", "dbb"]
    model_name = "DAVE2v3.pt"
    model = torch.load(f"../models/weights/{model_name}", map_location=device).eval()
    lossname = ""
    new_results_dir = ""
    sim = Simulator(
        scenario_name=default_scenario,
        spawnpoint_name=default_spawnpoint,
        path2sim=ARGS.path2src,
        steps_per_sec=15,
    )
    samples = 3
    newdir = "Experiment-{}-{}-{}".format(
        default_scenario, default_spawnpoint,
        "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)),
    )
    if not os.path.isdir("results/{}".format(newdir)):
        os.mkdir("results/{}".format(newdir))

    demo = Demo1()
    for cuton in cutons:
        sequence, unperturbed_results = demo.run_scenario_to_collect_sequence(sim, model, cuton, device=device)
        unperturbed_traj = unperturbed_results["traj"]


if __name__ == '__main__':
    main()