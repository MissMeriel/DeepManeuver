"""
.. module:: collect_images
    :platform: Windows
    :synopsis: Collect samples of runs and their outcomes

.. moduleauthor:: Meriel von Stein <meriel@virginia.edu>

"""
# import statistics
# import sys, time
# import os
# from matplotlib import pyplot as plt
# from matplotlib.pyplot import imshow
# from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject
# from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
# from scipy.spatial.transform import Rotation as R
# from scipy import interpolate
# from ast import literal_eval
# import logging, string, shutil
# from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, ProceduralCylinder, ProceduralCone, ProceduralBump, ProceduralRing
# from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer
# from superdeepbillboard.deepbillboard import DeepBillboard, SuperDeepBillboard
# import torch
# import cv2
# from skimage import util
# from PIL import Image
# from sklearn.metrics import mean_squared_error
# import kornia
# from torchvision.utils import save_image
# import pandas as pd
# import pickle
# from shapely.geometry import Polygon

import numpy as np
from matplotlib import pyplot as plt
import random, copy

from beamngpy import StaticObject, ScenarioObject, ProceduralCube

import math
from scipy.spatial.transform import Rotation as R
from ast import literal_eval
from scipy import interpolate


# MAKE SURE TO PULL METHODS FROM COLLECT_DATASET
class Simulator():

    def __init__(self, scenario="industrial", spawnpoint='racetrackstartinggate', swerving=None, swervingstdev=None,
                 expected_traj_file="DAVE2v1-lap-trajectory.txt"):
        self.scenario = scenario #'hirochi_raceway' #'industrial' #
        self.spawnpoint = spawnpoint #'startingline' #'driftcourse2' #'racetrackstartinggate' #
        self.swerving = swerving
        self.swervingstdev = swervingstdev
        self.remaining_centerline = []
        self.centerline = []
        self.centerline_interpolated = []
        self.roadleft = []
        self.roadright = []
        self.landwidth = 3.75  # 2.25
        self.roadwidth = 9.0 # 8.0
        self.qr_positions = []
        self.expected_trajectory = self.intake_lap_file(expected_traj_file)

    def get_spawnpoint(self):
        if self.scenario == 'automation_test_track':
            if self.spawnpoint == "startingline":
                return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
            elif self.spawnpoint == "startingline30mdown":
                # 30m down track from starting line
                return {'pos': (530.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
            elif self.spawnpoint == "handlingcircuit":
                return {'pos': (-294.031, 10.4074, 118.518), 'rot': None, 'rot_quat': (0, 0, 0.708103, 0.706109)}
            elif self.spawnpoint == "rallytrack":
                return {'pos': (-374.835, 84.8178, 115.084), 'rot': None, 'rot_quat': (0, 0, 0.718422, 0.695607)}
            elif self.spawnpoint == "highway":
                return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': (0, 0, -0.704635, 0.70957)}
            elif self.spawnpoint == "default":
                return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif self.scenario == 'hirochi_raceway':
            if self.spawnpoint =="figure8":
                # figure 8 oval
                return {'pos': (181.513, 4.62607, 20.6226), 'rot': None, 'rot_quat': (0, 0, 0.432016, 0.901866)}
            elif self.spawnpoint == "pitlane":
                return {'pos': (-457.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            elif self.spawnpoint == "paddock":
                return {'pos': (-256.046, 273.232, 25.1961), 'rot': None, 'rot_quat': (0, 0, 0.741246, 0.671234)}
            elif self.spawnpoint == "startingline":
                # return {'pos': (-408.48, 260.232, 25.4231), 'rot': None, 'rot_quat': (0, 0, -0.279907, 0.960027)}
                return {'pos': (-158.52922058,  187.46072006,   31.9925456), 'rot': None, 'rot_quat': (0.00075212895171717, 0.0062909661792219, 0.97648817300797, 0.21547800302505)}
            elif self.spawnpoint == "rockcrawl":
                # rock crawling course
                return {'pos': (-179.674, -50.6751, 27.6237), 'rot': None, 'rot_quat': (0.0734581, 0.00305369, 0.0414223, 0.996433)}
                #return {'pos': (-183.674, -38.6751, 25.6237), 'rot': None, 'rot_quat': (0.0734581, 0.0305369, 0.0414223, 0.996433)}
            elif self.spawnpoint == "default":
                return {'pos': (-453.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        elif self.scenario == 'utah':
            if self.spawnpoint == "westhighway":
                # return {'pos': (-922.158, -929.868, 135.534), 'rot': None, 'rot_quat': turn_180((0, 0, -0.820165, 0.572127))}
                return {'pos': (-1005.94, -946.995, 135.624), 'rot': None,
                        'rot_quat': (-0.0087888045236468, -0.0071660503745079, 0.38833409547806, 0.9214488863945)}
            elif self.spawnpoint == "westhighway1":
                return {'pos': (-151.542, -916.292, 134.561), 'rot': None, 'rot_quat': (0.017533652484417, 0.01487538497895, -0.68549990653992, 0.72770953178406)}
            if self.spawnpoint == "westhighway2":
                # after tunnel
                # return {'pos': (980.236, -558.879, 148.511), 'rot': None, 'rot_quat': (-0.015679769217968, -0.0069956826046109, 0.59496110677719, 0.80357110500336)}
                # return {'pos': (-241.669, -1163.42, 150.153), 'rot': None, 'rot_quat': (-0.0054957182146609, -0.0061398106627166, -0.69170582294464, 0.72213244438171)}
                return {'pos': (806.49749756, -652.42816162, 147.92123413), 'rot': None,
                        'rot_quat': (-0.0052490886300802, 0.007554049603641, 0.48879739642143, 0.87234884500504)}
            elif self.spawnpoint == "westhighway3":
                return {'pos': (-151.542, -916.292, 134.561), 'rot': None, 'rot_quat': (0.017533652484417, 0.01487538497895, -0.68549990653992, 0.72770953178406)}
            elif self.spawnpoint == "buildingsite":
                return {'pos': (-910.372, 607.927, 265.059), 'rot': None, 'rot_quat': (0, 0, 0.913368, -0.407135)}
            elif self.spawnpoint == "near_buildingsite":
                # on road near building site
                return {'pos': (-881.524, 611.674, 264.266), 'rot': None, 'rot_quat': (0, 0, 0.913368, -0.407135)}
            elif self.spawnpoint == "tourist_area":
                return {'pos': (-528.44, 283.886, 298.365), 'rot': None, 'rot_quat': (0, 0, 0.77543, 0.631434)}
            elif self.spawnpoint == "auto_repair_zone":
                return {'pos': (771.263, -149.268, 144.291), 'rot': None, 'rot_quat': (0, 0, -0.76648, 0.642268)}
            elif self.spawnpoint == "campsite":
                return {'pos': (566.186, -530.957, 135.291), 'rot': None, 'rot_quat': ( -0.0444918, 0.0124419, 0.269026, 0.962024)}
            elif self.spawnpoint == "default":
                return {'pos': ( 771.263, -149.268, 144.291), 'rot': None, 'rot_quat': (0, 0, -0.76648, 0.642268)} #(do not use for training)
                #COLLECTED UTAH8 return {'pos': (835.449, -164.877, 144.57), 'rot': None, 'rot_quat': (-0.003, -0.0048, -0.172, 0.985)}
            elif self.spawnpoint == "parkinglot":
                # parking lot (do not use for training)
                #return {'pos': (907.939, 773.502, 235.878), 'rot': None, 'rot_quat': (0, 0, -0.652498, 0.75779)} #(do not use for training)
                return {'pos': (963.22,707.785,235.583), 'rot': None, 'rot_quat': (-0.027, 0.018, -0.038, 0.999)}
        elif self.scenario == 'industrial':
            if self.spawnpoint == 'west':
            # western industrial area -- didnt work with AI Driver
                return {'pos': (237.131, -379.919, 34.5561), 'rot': None, 'rot_quat': (-0.035, -0.0181, 0.949, 0.314)}
            # open industrial area -- didnt work with AI Driver
            # drift course (dirt and paved)
            elif self.spawnpoint == 'driftcourse':
                return {'pos':(20.572, 161.438, 44.2149), 'rot': None, 'rot_quat': (-0.003, -0.005, -0.636, 0.771)}
            # drift course (all paved)
            elif self.spawnpoint == 'driftcourse2':
                return {'pos':(-177.25096130371094, 77.07855987548828, 42.56), 'rot': None, 'rot_quat': (-0.00086017715511844, -0.0085507640615106, 0.34327498078346, 0.93919563293457)}
            # rallycross course/default
            elif self.spawnpoint == 'rallycross':
                return {'pos': (4.85287, 160.992, 44.2151), 'rot': None, 'rot_quat': (-0.0032, 0.003, 0.763, 0.646)}
            # racetrack
            elif self.spawnpoint == 'racetrackright':
                return {'pos':(184.983, -41.0821, 42.7761), 'rot': None, 'rot_quat':(-0.005, 0.001, 0.299, 0.954)}
            elif self.spawnpoint == 'racetrackleft':
                return {'pos':(216.578, -28.1725, 42.7788), 'rot': None, 'rot_quat':(-0.0051003773696721, -0.0031468099914491, -0.67134761810303, 0.74111843109131)}
            elif self.spawnpoint == 'racetrackstartinggate':
                return {'pos':(185.84919357, -68.15278053,  42.56134033), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
                # return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
                # return {'pos':(216.775, -209.382, 45.4226), 'rot': None, 'rot_quat':(0.0031577029731125, 0.023062458261847, 0.99948292970657, 0.022182803601027)} #after hairpin turn
                # return {'pos': (271.936, -205.19, 44.0371), 'rot': None, 'rot_quat': (-0.014199636876583, 0.0051083415746689, 0.4541027545929, 0.89082151651382)}  # before hairpin turn
                # return {'pos': (241.368, -228.844, 45.1149), 'rot': None, 'rot_quat': (0.009736854583025, 0.0039774458855391, 0.24705672264099, 0.9689439535141)}  # before hairpin turn
            # racetrack sequence starting points
            if self.spawn_point == "curve1":
                # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.0022243109997362,0.0038272379897535,0.92039221525192,-0.39097133278847)}
                return {'pos': (210.314, -44.7132, 42.7758), 'rot': None,
                        'rot_quat': (0.0020199827849865, 0.0049774856306612, 0.92020887136459, -0.3913908302784)}
            elif self.spawn_point == "straight1":
                # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.0022243109997362,0.0038272379897535,0.92039221525192,-0.39097133278847)}
                # 130 steps
                # return {'pos': (252.028,-24.7376,42.814), 'rot': None,'rot_quat': (-0.044106796383858,0.05715386942029,-0.49562504887581,0.8655309677124)}
                # 50 steps
                return {'pos': (257.414, -27.9716, 42.8266), 'rot': None,
                        'rot_quat': (-0.032358665019274, 0.05354256555438, -0.45097458362579, 0.89034152030945)}
                # 4 steps
                return {'pos': (265.087, -33.7904, 42.805), 'rot': None,
                        'rot_quat': (-0.022659547626972, 0.023112617433071, -0.42281490564346, 0.90563786029816)}
            elif self.spawn_point == "curve2":
                # return {'pos':(323.432,-92.7588,43.6475), 'rot': None, 'rot_quat':(0.0083266003057361,0.013759891502559,-0.36539402604103,0.93071401119232)}
                return {'pos': (331.169, -104.166, 44.142), 'rot': None,
                        'rot_quat': (0.0095777017995715, 0.033657912164927, -0.35943350195885, 0.93251436948776)}
            # elif spawn_point == "straight2":
            # elif spawn_point == "curve3":
            # elif spawn_point == "straight3":
            # elif spawn_point == "curve4":
            elif self.spawn_point == "curve4":  # beginning of big curve right before starting gate
                # 50kph
                # return {'pos': (171.522, -109.746, 44.8168), 'rot': None,
                #         'rot_quat': (0.02015926875174, -0.023340219631791, 0.18135184049606, 0.982934653759)}
                # return {'pos': (169.177, -115.208, 44.4244), 'rot': None,
                #         'rot_quat': (0.022967331111431, -0.019803000614047, 0.19603483378887, 0.98012799024582)}
                # 40kph
                return {'pos': (166.291, -124.855, 43.7002), 'rot': None,
                        'rot_quat': (-0.023691736161709, -0.028266951441765, 0.14190191030502, 0.989193379879)}
        elif self.scenario == 'derby':
            if self.spawnpoint == 'big8':
                return {'pos': (-174.882, 61.4717, 83.5583), 'rot': None, 'rot_quat': (-0.119, -0.001, 0.002, 0.993)}
        elif self.scenario == 'east_coast_usa':
            if self.spawnpoint == "town_industrial_area":
                return {'pos':(736.768, -20.8732, 52.127), 'rot': None, 'rot_quat':(-0.006, -0.004, -0.186, 0.983)}
            elif self.spawnpoint == "farmhouse":
                return {'pos':(-607.898, -354.424, 34.5097), 'rot': None, 'rot_quat':(-0.0007, 0.0373, 0.960, -0.279)}
            elif self.spawnpoint == "gas_station_lot":
                return {'pos':(-758.764, 480.25, 23.774), 'rot': None, 'rot_quat':(-0.001, -0.010, -0.739, 0.673)}
            elif self.spawnpoint == "sawmill":
                return {'pos':(261.326, -774.902, 46.2887), 'rot': None, 'rot_quat':(-0.005, 0.008, 0.950, -0.311)}
            elif self.spawnpoint == "default":
                # highway/default
                return {'pos':(900.643, -226.266, 40.191), 'rot': None, 'rot_quat':(-0.004, -0.0220, -0.0427, 0.99)}
        elif self.scenario == 'driver_training': #etk driver experience center
            if self.spawnpoint == 'north':
                return {'pos':(-195.047, 253.654, 53.019), 'rot': None, 'rot_quat':(-0.006, -0.006, -0.272, 0.962)}
            elif self.spawnpoint == ' west':
                return {'pos': (-394.541, 69.052, 51.2327), 'rot': None, 'rot_quat': (-0.0124, 0.0061, -0.318, 0.948)}
            elif self.spawnpoint == ' default':
                return {'pos':(60.6395, 70.8329, 38.3048), 'rot': None, 'rot_quat':(0.015, 0.006, 0.884, 0.467)}
                #return {'pos': (32.3209, 89.8991, 39.135), 'rot': None, 'rot_quat': (0.0154, -0.007, 0.794, 0.607)}
        elif self.scenario == 'jungle_rock_island':
            if self.spawnpoint == 'industrialsite':
                #return {'pos': (38.0602, 559.695, 156.726), 'rot': None, 'rot_quat': (-0.004, 0.005, 0.1, -0.005)}
                return {'pos': (-9.99082, 580.726, 156.72), 'rot': None, 'rot_quat': (-0.0066664000041783, 0.0050910739228129, 0.62305396795273, 0.78213387727737)}
            elif self.spawnpoint == 'turning_radius_test':
                # return {'pos': (49.9264, 535.766, 156.725), 'rot': None, 'rot_quat': (-0.001217813231051, 0.0052481382153928, 0.96666145324707, 0.2560011446476)}
                return {'pos': (53.5337, 527.549, 156.906), 'rot': None, 'rot_quat': (0.0032839819323272, 0.0023067169822752, 0.99960005283356, -0.027994954958558)}
            elif self.spawnpoint == 'deceleration_test':
                return {'pos': (47.6473, 532.276, 156.922), 'rot': None, 'rot_quat': (0.0031495571602136, 0.0068701906129718, 0.99776315689087, 0.066419698297977)}
            elif self.spawnpoint == 'observatory':
                return {'pos':(-842.505, 820.688, 186.139), 'rot': None, 'rot_quat':(0.0003, 0.0122, 0.994, 0.113)}
            elif self.spawnpoint == 'hangar':
                return {'pos':(818.098, -676.55, 160.034), 'rot': None, 'rot_quat':(-0.027693340554833, 0.011667124927044, -0.19988858699799, 0.97935771942139)}
            elif self.spawnpoint == 'peninsula':
                return {'pos':(355.325, -775.203, 133.412), 'rot': None, 'rot_quat':(0.0243, -0.0422, -0.345, 0.937)}
            elif self.spawnpoint == 'port':
                return {'pos':(-590.56, 312.523, 130.215), 'rot': None, 'rot_quat':(-0.0053834365680814, 0.00023860974761192, 0.013710686005652, 0.99989157915115)}
            elif self.spawnpoint == 'default':
                # hill/default
                return {'pos':(124.232, -78.7489, 158.735), 'rot': None, 'rot_quat':(0.005, 0.0030082284938544, 0.96598142385483, 0.25854349136353)}
        elif self.scenario == 'small_island':
            if self.spawnpoint == 'default':
                # north road/default
                return {'pos': (254.77, 233.82, 39.5792), 'rot': None, 'rot_quat': (-0.013, 0.008, -0.0003, 0.1)}
            elif self.spawnpoint == 'southroad':
                return {'pos':(-241.908, -379.478, 31.7824), 'rot': None, 'rot_quat':(0.008, 0.006, -0.677, 0.736)}
            elif self.spawnpoint == 'industrialarea':
                return {'pos':(126.907, 272.76, 40.0575), 'rot': None, 'rot_quat':(-0.0465, -0.0163, -0.0546, 0.997)}
        elif self.scenario == 'smallgrid':
            if self.spawnpoint == 'default':
                return {'pos': (0.0, 0.0, 0.217252), 'rot': None, 'rot_quat': self.turn_X_degrees((-0.0055627916008234, 0.00013624821440317, -6.5445169639133e-07, 0.999984562397), degrees=270)}

    def turn_X_degrees(self, rot_quat, degrees=90):
        r = R.from_quat(list(rot_quat))
        r = r.as_euler('xyz', degrees=True)
        r[2] = r[2] + degrees
        r = R.from_euler('xyz', r, degrees=True)
        return tuple(r.as_quat())

    def lineseg_dists(self, p, a, b):
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

    # def turn_90(self, rot_quat):
    #     r = R.from_quat(list(rot_quat))
    #     r = r.as_euler('xyz', degrees=True)
    #     r[2] = r[2] + 90
    #     r = R.from_euler('xyz', r, degrees=True)
    #     return tuple(r.as_quat())

    def rpy_from_vec(self, vec):
        r = R.from_rotvec(list(vec))
        r = r.as_euler('xyz', degrees=True)
        return r

    #return distance between two 3d points
    def distance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def distance2D(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def add_barriers(self, scenario):
        barrier_locations = []
        with open('industrial_racetrack_barrier_locations.txt', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(' ')
                pos = line[0].split(',')
                pos = tuple([float(i) for i in pos])
                rot_quat = line[1].split(',')
                rot_quat = tuple([float(j) for j in rot_quat])
                rot_quat = self.turn_X_degrees(rot_quat, degrees=90)
                ramp = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                    shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
                scenario.add_object(ramp)

    def add_qr_cubes(self, scenario, qr_box_file):
        self.qr_positions = []
        with open('posefiles/qr_box_locations.txt', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(' ')
                pos = line[0].split(',')
                pos = tuple([float(i) for i in pos])
                rot_quat = line[1].split(',')
                rot_quat = tuple([float(j) for j in rot_quat])
                self.qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
                box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
                                     rot_quat=rot_quat, scale=(5, 5, 5), JBeam='qrbox2', datablock="default_vehicle")
                scenario.add_object(box)
            if self.scenario_id == "industrial" and self.spawnpoint_id == "curve1":
                cube = ProceduralCube(name='cube_platform',
                                      pos=(145.214, -160.72, 43.7269),
                                      rot=None,
                                      rot_quat=(0, 0, 0, 1),
                                      size=(2, 6, 0.5))
                scenario.add_procedural_mesh(cube)
            elif self.scenario_id == "driver_training" and self.spawnpoint_id == "approachingfork":
                cube = ProceduralCube(name='cube_platform',
                                      pos=(-20.3113, 218.448, 50.043),
                                      rot=None,
                                      rot_quat=(
                                      -0.022064134478569, -0.022462423890829, 0.82797580957413, 0.55987912416458),
                                      size=(4, 8, 0.5))
                scenario.add_procedural_mesh(cube)

    def from_val_get_key(self, d, v):
        key_list = list(d.keys())
        val_list = list(d.values())
        position = val_list.index(v)
        return key_list[position]

    def plot_racetrack_roads(self, roads, bng):
        global default_scenario, spawnpoint
        colors = ['b','g','r','c','m','y','k']
        symbs = ['-','--','-.',':','.',',','v','o','1',]
        for road in roads:
            road_edges = bng.get_road_edges(road)
            x_temp = []
            y_temp = []
            dont_add = False
            xy_def = [edge['middle'][:2] for edge in road_edges]
            dists = [self.distance2D(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
            if sum(dists) > 500 and (road != "9096" or road != "9206"):
                continue
            for edge in road_edges:
                # if edge['middle'][1] < 0:
                #     dont_add=True
                #     break
                # if edge['middle'][0] > -75:
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
        plt.scatter(-408.48, 260.232, marker="o", linewidths=10, label="spawnpoint")
        plt.legend()
        plt.title("{} {}".format(default_scenario, spawnpoint))
        plt.show()
        plt.pause(0.001)

    def road_analysis(self, bng, road_id):
        # self.plot_racetrack_roads(bng.get_roads(), bng)
        # get relevant road
        edges = []
        adjustment_factor = 4.0
        industrial_racetrack_pts = ["startinggate", "straight1", "straight2", "curve2"]
        if self.scenario == "industrial" and self.spawnpoint in industrial_racetrack_pts:
            edges = bng.get_road_edges(road_id)
            adjustment_factor = 10
        elif self.scenario == "industrial" and self.spawnpoint == "driftcourse2":
            edges = bng.get_road_edges('7987')
        elif self.scenario == "hirochi_raceway" and self.spawnpoint == "startingline":
            edges = bng.get_road_edges('9096')
            edges.extend(bng.get_road_edges('9206'))
            # edges = bng.get_road_edges('9206')
        elif self.scenario == "utah" and self.spawnpoint == "westhighway":
            edges = bng.get_road_edges('15145')
            # edges.extend(bng.get_road_edges('15162'))
            edges.extend(bng.get_road_edges('15154'))
            edges.extend(bng.get_road_edges('15810'))
            edges.extend(bng.get_road_edges('16513'))
            adjustment_factor = 1.0
        elif self.scenario == "utah" and self.spawnpoint == "westhighway2":
            edges = bng.get_road_edges('15810')
            # edges.extend(bng.get_road_edges('15810'))
            edges.extend(bng.get_road_edges('16513'))
            # edges.extend(bng.get_road_edges('15143'))
            # edges = bng.get_road_edges('9206')
            adjustment_factor = 1.0
        elif self.scenario == "utah" and self.spawnpoint == "undef":
            edges = bng.get_road_edges('15852')
            edges.extend(bng.get_road_edges('14904'))
            edges.extend(bng.get_road_edges('15316'))
            adjustment_factor = 1.0
        elif self.scenario == "driver_training" and self.spawnpoint == "approachingfork":
            edges = bng.get_road_edges("7719")
            edges.reverse()
            adjustment_factor = -0.3
            # adjustment_factor = -0.01
            for i in range(len(edges)):
                #     edges[i]['left'] = np.array(edges[i]['middle']) + (np.array(edges[i]['left']) - np.array(edges[i]['middle']))/ -0.1
                edges[i]['right'] = np.array(edges[i]['middle']) + (
                            np.array(edges[i]['right']) - np.array(edges[i]['middle'])) / 0.1
            # edges = bng.get_road_edges('7936')
            # edges.extend(bng.get_road_edges('7836')) #7952
        print("retrieved road edges")
        self.actual_middle = [edge['middle'] for edge in edges]
        self.roadmiddle = copy.deepcopy(self.actual_middle)
        print(f"{self.actual_middle[0]=}")
        self.roadleft = [edge['left'] for edge in edges]
        self.roadright = [edge['right'] for edge in edges]
        self.adjusted_middle = [
            np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle'])) / adjustment_factor for edge
            in edges]
        self.centerline = self.actual_middle
        return self.actual_middle, self.adjusted_middle, self.roadleft, self.roadright

    def create_ai_line_from_road_with_interpolation(self, spawn, bng, swerving=False):
        # global centerline, remaining_centerline, centerline_interpolated
        points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
        actual_middle, adjusted_middle = self.road_analysis(bng)
        print("finished road analysis")
        middle_end = adjusted_middle[:3]
        middle = adjusted_middle[3:]
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
                dists.append(self.distance2D(p[:-1], middle[i+1][:-1]))
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
                if self.distance(p, middle[i+1]) > 1:
                    y_interp = interpolate.interp1d([p[0], middle[i + 1][0]], [p[1], middle[i + 1][1]])
                    num = int(self.distance(p, middle[i + 1]))
                    xs = np.linspace(p[0], middle[i + 1][0], num=num, endpoint=True)
                    ys = y_interp(xs)
                    traj.extend([[x,y] for x,y in zip(xs,ys)])
                else:
                    traj.append([p[0],p[1]])
            # interpolate swerve line and swerve flags
            for i,p in enumerate(swerving_middle[:-1]):
                # interpolate at 1m distance
                if self.distance(p, swerving_middle[i+1]) > 1:
                    y_interp = interpolate.interp1d([p[0], swerving_middle[i + 1][0]], [p[1], swerving_middle[i + 1][1]])
                    num = int(self.distance(p, swerving_middle[i + 1]))
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
                if self.distance(p, middle[i+1]) > 1:
                    y_interp = interpolate.interp1d([p[0], middle[i + 1][0]], [p[1], middle[i + 1][1]])
                    num = int(self.distance(p, middle[i + 1]))
                    xs = np.linspace(p[0], middle[i + 1][0], num=num, endpoint=True)
                    ys = y_interp(xs)
                    traj.extend([[x,y,p[2]] for x,y in zip(xs,ys)])
                else:
                    traj.append(copy.deepcopy(p))
        print("set up debug line")
        # set up debug line
        for i,p in enumerate(actual_middle[:-1]):
            points.append([p[0], p[1], p[2]])
            point_colors.append([0, 1, 0, 0.1])
            spheres.append([p[0], p[1], p[2], 0.25])
            sphere_colors.append([1, 0, 0, 0.8])
        print("spawn point:{}".format(spawn))
        print("beginning of script:{}".format(middle[0]))
        self.plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
        centerline = copy.deepcopy(traj)
        remaining_centerline = copy.deepcopy(traj)
        centerline_interpolated = copy.deepcopy(traj)
        # for i in range(4):
        #     centerline.extend(copy.deepcopy(centerline))
        #     remaining_centerline.extend(copy.deepcopy(remaining_centerline))
        bng.add_debug_line(points, point_colors,
                           spheres=spheres, sphere_colors=sphere_colors,
                           cling=True, offset=0.1)
        return (points, point_colors, spheres, sphere_colors, True, 0.1)

    def find_width_of_road(self, bng):
        edges = bng.get_road_edges('7983')
        left_edge = [edge['left'] for edge in edges]
        right_edge = [edge['right'] for edge in edges]
        middle = [edge['middle'] for edge in edges]
        dist1 = self.distance(left_edge[0], middle[0])
        dist2 = self.distance(right_edge[0], middle[0])
        print("width of road:", (dist1 + dist2))
        return dist1 + dist2

    def returned_to_expected_traj(self, pos_window):
        global expected_trajectory
        dists = []
        for point in pos_window:
            dist = self.dist_from_line(expected_trajectory, point)
            dists.append(min(dist))
        avg_dist = sum(dists) / len(dists)
        return avg_dist < 1


    def dist_from_line(self, centerline, point):
        a = [[x[0],x[1]] for x in centerline[:-1]]
        b = [[x[0],x[1]] for x in centerline[1:]]
        a = np.array(a)
        b = np.array(b)
        dist = self.lineseg_dists([point[0], point[1]], a, b)
        return dist


    def intake_lap_file(self, filename="DAVE2v1-lap-trajectory.txt"):
        expected_trajectory = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                line = literal_eval(line)
                expected_trajectory.append(line)
        return expected_trajectory


    # track is approximately 12.50m wide
    # car is approximately 1.85m wide
    def has_car_left_track(self, vehicle_pos, vehicle_bbox, bng):
        global centerline_interpolated
        # get nearest road point
        distance_from_centerline = self.dist_from_line(centerline_interpolated, vehicle_pos)
        # check if it went over left edge
        # print("Distance from center of road:", min(distance_from_centerline))
        return min(distance_from_centerline) > 15.0


    def laps_completed(self, lapcount):
        global centerline_interpolated, remaining_centerline
        remainder = centerline_interpolated.index(remaining_centerline[0])
        remainder = remainder / len(centerline_interpolated)
        return lapcount + remainder


    def adjust_centerline_for_spawn(self):
        return


    def setup_in_opposite_direction(self):
        return