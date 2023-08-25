#!/bin/bash
bbsizes=[5, 10, 15]
iterations=[400]
noiselevels=[10, 15, 20, 1000]
rscs=[0.60]
cutons=[20, 24, 28]
input_divs=[False, True]
direction=["left", "right"]
techniques=["deepman", "dbb-orig", "dbb"]
samples=50
path2src="../.."
python3 collect-perturbed-trace-metas-demo1.py $path2src  straight1