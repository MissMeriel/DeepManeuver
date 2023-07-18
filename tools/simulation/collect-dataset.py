"""
.. module:: collect_images
    :platform: Windows
    :synopsis: Collect samples of runs and their outcomes

.. moduleauthor:: Meriel von Stein <meriel@virginia.edu>

"""
import random, math
import sys, time
import numpy as np
import os, copy
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from ast import literal_eval
import logging
import string
import shutil

# globals
default_scenario = 'driver_training' #'industrial'
spawnpoint = 'approachingfork'
numsamps = 25000
collection_hash = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
training_dir = 'H:/BeamNG_DeepBillboard_dataset2/training_runs_{}-{}_{}samples-{}deletelater6/'.format(default_scenario, spawnpoint, numsamps, collection_hash)
default_model = 'hopper'
overall_throttle_setpoint = 40
throttle_curve_setpt = 30
swerving = None
swervingstdev = None
throttle_setpoint = overall_throttle_setpoint
integral=0; prev_error=0
steer_integral=0; steer_prev_error=0; steer_prev_setpoint = 0
remaining_centerline = []
centerline = []
centerline_interpolated = []
roadleft = []
roadright = []
avg_error = []
dt = 20
#base_filename = '{}/{}/{}_{}_'.format(os.getcwd(), training_dir, default_model, default_scenario.replace("_", ""))
base_filename = '{}_{}_'.format(default_model, default_scenario.replace("_", ""))
steps_per_second = 30
tight_curve = False

def spawn_point(scenario_locale, spawn_point):
    if scenario_locale == 'automation_test_track':
        # starting line
        #return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        # 30m down track from starting line
        #return {'pos': (530.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        # handling circuit
        #return {'pos': (-294.031, 10.4074, 118.518), 'rot': None, 'rot_quat': (0, 0, 0.708103, 0.706109)}
        # rally track
        #return {'pos': (-374.835, 84.8178, 115.084), 'rot': None, 'rot_quat': (0, 0, 0.718422, 0.695607)}
        # highway
        return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': (0, 0, -0.704635, 0.70957)}
        # default
        return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
    elif scenario_locale == 'hirochi_raceway':
        if spawnpoint =="figure8":
        # figure 8 oval
            return {'pos': (181.513, 4.62607, 20.6226), 'rot': None, 'rot_quat': (0, 0, 0.432016, 0.901866)}
        elif spawnpoint == "pitlane":
            return {'pos': (-457.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        elif spawnpoint == "paddock":
            return {'pos': (-256.046, 273.232, 25.1961), 'rot': None, 'rot_quat': (0, 0, 0.741246, 0.671234)}
        elif spawnpoint == "startingline":
            # return {'pos': (-408.48, 260.232, 25.4231), 'rot': None, 'rot_quat': (0, 0, -0.279907, 0.960027)}
            return {'pos': (-158.52922058,  187.46072006,   31.9925456), 'rot': None, 'rot_quat': (0.00075212895171717, 0.0062909661792219, 0.97648817300797, 0.21547800302505)}
        # 172.573 | E | libbeamng.lua.V.updateGFX | Object position: vec3(-152.69, 182.443, 32.4299)
        # 172.575 | E | libbeamng.lua.V.updateGFX | Object rotation(quat): quat(0.00075212895171717, 0.0062909661792219, 0.97648817300797, 0.21547800302505)
        elif spawnpoint == "rockcrawl":
            # rock crawling course
            return {'pos': (-179.674, -50.6751, 27.6237), 'rot': None, 'rot_quat': (0.0734581, 0.00305369, 0.0414223, 0.996433)}
            #return {'pos': (-183.674, -38.6751, 25.6237), 'rot': None, 'rot_quat': (0.0734581, 0.0305369, 0.0414223, 0.996433)}
        elif spawnpoint == "default":
            return {'pos': (-453.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
    elif scenario_locale == 'utah':
        if spawnpoint == "westhighway":
            # return {'pos': (-922.158, -929.868, 135.534), 'rot': None, 'rot_quat': turn_180((0, 0, -0.820165, 0.572127))}
            return {'pos': (-1005.94, -946.995, 135.624), 'rot': None, 'rot_quat': (-0.0087888045236468, -0.0071660503745079, 0.38833409547806, 0.9214488863945)}
        # west highway 2
        #COLLECTED UTAH10 return {'pos': (-151.542, -916.292, 134.561), 'rot': None, 'rot_quat': (0.017533652484417, 0.01487538497895, -0.68549990653992, 0.72770953178406)}
        if spawnpoint == "westhighway2":
            # after tunnel
            # return {'pos': (980.236, -558.879, 148.511), 'rot': None, 'rot_quat': (-0.015679769217968, -0.0069956826046109, 0.59496110677719, 0.80357110500336)}
            # return {'pos': (-241.669, -1163.42, 150.153), 'rot': None, 'rot_quat': (-0.0054957182146609, -0.0061398106627166, -0.69170582294464, 0.72213244438171)}
            return {'pos': (806.49749756, -652.42816162,  147.92123413), 'rot': None, 'rot_quat': (-0.0052490886300802, 0.007554049603641, 0.48879739642143, 0.87234884500504)}
        # building site
        #return {'pos': (-910.372, 607.927, 265.059), 'rot': None, 'rot_quat': (0, 0, 0.913368, -0.407135)}
        # on road near building site
        #COLLECTED UTAH7 #return {'pos': (-881.524, 611.674, 264.266), 'rot': None, 'rot_quat': (0, 0, 0.913368, -0.407135)}
        # tourist area
        #return {'pos': (-528.44, 283.886, 298.365), 'rot': None, 'rot_quat': (0, 0, 0.77543, 0.631434)}
        # auto repair zone
        #return {'pos': (771.263, -149.268, 144.291), 'rot': None, 'rot_quat': (0, 0, -0.76648, 0.642268)}
        # campsite
        #return {'pos': (566.186, -530.957, 135.291), 'rot': None, 'rot_quat': ( -0.0444918, 0.0124419, 0.269026, 0.962024)}
        # default
        #return {'pos': ( 771.263, -149.268, 144.291), 'rot': None, 'rot_quat': (0, 0, -0.76648, 0.642268)} #(do not use for training)
        #COLLECTED UTAH8 return {'pos': (835.449, -164.877, 144.57), 'rot': None, 'rot_quat': (-0.003, -0.0048, -0.172, 0.985)}
        # parking lot (do not use for training)
        #return {'pos': (907.939, 773.502, 235.878), 'rot': None, 'rot_quat': (0, 0, -0.652498, 0.75779)} #(do not use for training)
        #COLLECTED UTAH9 return {'pos': (963.22,707.785,235.583), 'rot': None, 'rot_quat': (-0.027, 0.018, -0.038, 0.999)}
    elif scenario_locale == 'industrial':
        if spawnpoint == 'west':
        # western industrial area -- didnt work with AI Driver
            return {'pos': (237.131, -379.919, 34.5561), 'rot': None, 'rot_quat': (-0.035, -0.0181, 0.949, 0.314)}
        # open industrial area -- didnt work with AI Driver
        # drift course (dirt and paved)
        elif spawnpoint == 'driftcourse':
            return {'pos':(20.572, 161.438, 44.2149), 'rot': None, 'rot_quat': (-0.003, -0.005, -0.636, 0.771)}
        # drift course (all paved)
        elif spawnpoint == 'driftcourse2':
            return {'pos':(-177.25096130371094, 77.07855987548828, 42.56), 'rot': None, 'rot_quat': (-0.00086017715511844, -0.0085507640615106, 0.34327498078346, 0.93919563293457)}
        # rallycross course/default
        elif spawnpoint == 'rallycross':
            return {'pos': (4.85287, 160.992, 44.2151), 'rot': None, 'rot_quat': (-0.0032, 0.003, 0.763, 0.646)}
        # racetrack
        elif spawnpoint == 'racetrackright':
            return {'pos':(184.983, -41.0821, 42.7761), 'rot': None, 'rot_quat':(-0.005, 0.001, 0.299, 0.954)}
        elif spawnpoint == 'racetrackleft':
            return {'pos':(216.578, -28.1725, 42.7788), 'rot': None, 'rot_quat':(-0.0051003773696721, -0.0031468099914491, -0.67134761810303, 0.74111843109131)}
        elif spawnpoint == 'racetrackstartinggate':
            return {'pos':(185.84919357, -68.15278053,  42.56134033), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
            # return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
            # return {'pos':(216.775, -209.382, 45.4226), 'rot': None, 'rot_quat':(0.0031577029731125, 0.023062458261847, 0.99948292970657, 0.022182803601027)} #after hairpin turn
            # return {'pos': (271.936, -205.19, 44.0371), 'rot': None, 'rot_quat': (-0.014199636876583, 0.0051083415746689, 0.4541027545929, 0.89082151651382)}  # before hairpin turn
            # return {'pos': (241.368, -228.844, 45.1149), 'rot': None, 'rot_quat': (0.009736854583025, 0.0039774458855391, 0.24705672264099, 0.9689439535141)}  # before hairpin turn
    elif scenario_locale == 'derby':
        # the big 8
        if spawnpoint == 'big8':
            return {'pos': (-174.882, 61.4717, 83.5583), 'rot': None, 'rot_quat': (-0.119, -0.001, 0.002, 0.993)}
    elif scenario_locale == 'east_coast_usa':
        # town industrial area
        #COLLECTED EAST_COAST_USA1 return {'pos':(736.768, -20.8732, 52.127), 'rot': None, 'rot_quat':(-0.006, -0.004, -0.186, 0.983)}
        # farmhouse
        #COLLECTED EAST_COAST_USA2 return {'pos':(-607.898, -354.424, 34.5097), 'rot': None, 'rot_quat':(-0.0007, 0.0373, 0.960, -0.279)}
        # gas station parking lot
        #COLLECTED EAST_COAST_USA3 return {'pos':(-758.764, 480.25, 23.774), 'rot': None, 'rot_quat':(-0.001, -0.010, -0.739, 0.673)}
        # sawmill
        #COLLECTED EAST_COAST_USA4 return {'pos':(261.326, -774.902, 46.2887), 'rot': None, 'rot_quat':(-0.005, 0.008, 0.950, -0.311)}
        if spawnpoint == 'default':
            # highway/default
            return {'pos':(900.643, -226.266, 40.191), 'rot': None, 'rot_quat':(-0.004, -0.0220, -0.0427, 0.99)}
    elif scenario_locale == 'driver_training': #etk driver experience center
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
    elif scenario_locale == 'jungle_rock_island':
        # industrial site -- weird behavior, redo
        #return {'pos': (38.0602, 559.695, 156.726), 'rot': None, 'rot_quat': (-0.004, 0.005, 0.1, -0.005)}
        return {'pos': (-9.99082, 580.726, 156.72), 'rot': None, 'rot_quat': (-0.0066664000041783, 0.0050910739228129, 0.62305396795273, 0.78213387727737)}
        # observatory
        # COLLECTED JUNGLE_ROCK_ISLAND2 return {'pos':(-842.505, 820.688, 186.139), 'rot': None, 'rot_quat':(0.0003, 0.0122, 0.994, 0.113)}
        # hangar
        # COLLECTED JUNGLE_ROCK_ISLAND3 return {'pos':(818.098, -676.55, 160.034), 'rot': None, 'rot_quat':(-0.027693340554833, 0.011667124927044, -0.19988858699799, 0.97935771942139)}
        # peninsula
        # COLLECTED JUNGLE_ROCK_ISLAND4 return {'pos':(355.325, -775.203, 133.412), 'rot': None, 'rot_quat':(0.0243, -0.0422, -0.345, 0.937)}
        # port
        # COLLECTED JUNGLE_ROCK_ISLAND5 return {'pos':(-590.56, 312.523, 130.215), 'rot': None, 'rot_quat':(-0.0053834365680814, 0.00023860974761192, 0.013710686005652, 0.99989157915115)}
        # hill/default
        if spawnpoint == 'default':
            return {'pos':(124.232, -78.7489, 158.735), 'rot': None, 'rot_quat':(0.005, 0.0030082284938544, 0.96598142385483, 0.25854349136353)}
    elif scenario_locale == 'small_island':
        if spawnpoint == 'default':
            # north road/default
            return {'pos': (254.77, 233.82, 39.5792), 'rot': None, 'rot_quat': (-0.013, 0.008, -0.0003, 0.1)}
        elif spawnpoint == 'southroad':
            # south road
            return {'pos':(-241.908, -379.478, 31.7824), 'rot': None, 'rot_quat':(0.008, 0.006, -0.677, 0.736)}
        elif spawnpoint == 'industrialarea':
            return {'pos':(126.907, 272.76, 40.0575), 'rot': None, 'rot_quat':(-0.0465, -0.0163, -0.0546, 0.997)}

def setup_sensors(vehicle, pos=(-0.5, 0.38, 1.3), direction=(0, 1.0, 0)):
    # Set up sensors
    # pos = (-0.3, 1, 1.0) # default
    # pos = (-0.5, 2, 1.0) #center edge of hood
    # pos = (-0.5, 1, 1.0)  # center middle of hood
    # pos = (-0.5, 0.4, 1.0)  # dashboard
    # pos = (-0.5, 0.38, 1.5) # roof
    # pos = (-0.5, 0.38, 1.3) # windshield
    # direction = (0, 1.0, 0)
    fov = 51 # 60 works for full lap #63 breaks on hairpin turn
    resolution = (400,225) #(200, 150) (320, 180) #(1280,960) #(512, 512)
    front_camera = Camera(pos, direction, fov, resolution,
                          colour=True, depth=True, annotation=True)
    # pos = pos
    # direction = (0, 1, 0)
    # fov = 120
    # resolution = (1280,960)
    # back_camera = Camera(pos, direction, fov, resolution,
    #                      colour=True, depth=True, annotation=True)

    gforces = GForces()
    electrics = Electrics()
    damage = Damage()
    #lidar = Lidar(visualized=False)
    timer = Timer()

    # Attach them
    vehicle.attach_sensor('front_cam', front_camera)
    # vehicle.attach_sensor('back_cam', back_camera)
    vehicle.attach_sensor('gforces', gforces)
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('damage', damage)
    vehicle.attach_sensor('timer', timer)
    return vehicle

def setup_dir(training_dir):
    #d = "{}/{}".format(os.path.dirname(os.path.realpath(__file__)), training_dir)
    training_basename = "/".join(training_dir.split("/")[:-1])
    if not os.path.exists(os.path.dirname(training_basename)):
        os.mkdir(os.path.dirname(training_basename))
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)
    return "{}/data.csv".format(training_dir)

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

def turn_90(rot_quat):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + 90
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())

def turn_180(rot_quat):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + 180
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())

def rpy_from_vec(vec):
    r = R.from_rotvec(list(vec))
    r = r.as_euler('xyz', degrees=True)
    return r

#return distance between two 3d points
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def distance2D(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

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

def from_val_get_key(d, v):
    key_list = list(d.keys())
    val_list = list(d.values())
    position = val_list.index(v)
    return key_list[position]

def throttle_PID(kph, dt):
    global integral, prev_error, throttle_setpoint
    # kp = 0.001; ki = 0.00001; kd = 0.0001
    # kp = .3; ki = 0.01; kd = 0.1
    # kp = 0.15; ki = 0.0001; kd = 0.008 # worked well but only got to 39kph
    kp = 0.19; ki = 0.0001; kd = 0.008
    error = throttle_setpoint - kph
    deriv = (error - prev_error) / dt
    integral = integral + error * dt
    w = kp * error + ki * integral + kd * deriv
    prev_error = error
    return w

def steering_PID(curr_steering,  steer_setpoint, dt):
    global steer_integral, steer_prev_error, steer_prev_setpoint
    global tight_curve, avg_error
    # kp = 1.1; ki = 0.05; kd = 0.001 #0.075
    if tight_curve:
        # kp = 50; ki = 0.0; kd = 0.1 #0.075
        print("returning ", 1 * np.sign(steer_setpoint), "\n")
        return 1 * np.sign(steer_setpoint)
    else:
        # kp = 1.1; ki = 0.05; kd = 0.001
        # kp = 1.7; ki = 1.1; kd = 0.0
        kp = 170; ki = 50; kd = 0.1 # works with 40-30 kph
        kp = 170; ki = 50; kd = 0.05 # experimental to handle curves slightly better
        # kp = 0.167; ki = 0.11; kd = 0.14
        # print("steer_setpoint", steer_setpoint)
        # print("curr_steering ", curr_steering)
        # print("tight_curve", tight_curve)
        # print("dt:", dt)
        error = steer_setpoint - curr_steering
        deriv = (error - steer_prev_error) / dt
        integral = steer_integral + error * dt
        w = kp * error + ki * integral + kd * deriv
        steer_prev_error = error
        # print("error term", kp * error)
        # print("deriv term", kd * deriv)
        # print("integral term", ki * integral)
        # print("steering w", w)
        # # print("return val", curr_steering + w)
        # # print("\n")
        # print("returning ", w+curr_steering, "\n")
    return w + curr_steering

def plot_racetrack_roads(roads, bng):
    global default_scenario, spawnpoint
    colors = ['b','g','r','c','m','y','k']
    symbs = ['-','--','-.',':','.',',','v','o','1',]
    print(f"{len(roads)=}")
    print(f"{roads.keys()=}")
    for road in roads:
        road_edges = bng.get_road_edges(road)
        x_temp = []
        y_temp = []
        dont_add = False
        xy_def = [edge['middle'][:2] for edge in road_edges]
        dists = [distance2D(xy_def[i], xy_def[i+1]) for i,p in enumerate(xy_def[:-1])]
        # if sum(dists) > 500 and (road != "9096" or road != "9206"):
        s = sum(dists)
        if (s < 250):
            continue
        for edge in road_edges:
            if edge['middle'][1] <0:
                dont_add=True
                break
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
    # plt.scatter(-922.158, -929.868, marker="o", linewidths=10, label="spawnpoint")
    sp = spawn_point(default_scenario, spawnpoint)
    plt.scatter(sp['pos'][0], sp['pos'][1], marker="o", linewidths=10, label="spawnpoint")
    # plt.legend()
    plt.legend(ncol=3, handleheight=2.4, labelspacing=0.05)
    plt.title("{} {}".format(default_scenario, spawnpoint))
    plt.savefig("{}/{}-{}-map.jpg".format(training_dir, default_scenario, spawnpoint))
    plt.show()
    plt.pause(0.001)

def road_analysis(bng):
    global centerline, roadleft, roadright
    global default_scenario, spawnpoint
    plot_racetrack_roads(bng.get_roads(), bng)
    # get relevant road
    edges = []
    adjustment_factor = 4.0
    if default_scenario == "industrial" and spawnpoint == "racetrackstartinggate":
        edges =bng.get_road_edges('7983')
    elif default_scenario == "industrial" and spawnpoint == "driftcourse2":
        edges = bng.get_road_edges('7987')
    elif default_scenario == "hirochi_raceway" and spawnpoint == "startingline":
        edges = bng.get_road_edges('9096')
        edges.extend(bng.get_road_edges('9206'))
        # edges = bng.get_road_edges('9206')
    elif default_scenario == "utah" and spawnpoint == "westhighway":
        edges = bng.get_road_edges('15145')
        # edges.extend(bng.get_road_edges('15162'))
        edges.extend(bng.get_road_edges('15154'))
        edges.extend(bng.get_road_edges('15810'))
        edges.extend(bng.get_road_edges('16513'))
        adjustment_factor = 1.0
    elif default_scenario == "utah" and spawnpoint == "westhighway2":
        edges = bng.get_road_edges('15810')
        # edges.extend(bng.get_road_edges('15810'))
        edges.extend(bng.get_road_edges('16513'))
        # edges.extend(bng.get_road_edges('15143'))
        # edges = bng.get_road_edges('9206')
        adjustment_factor = 1.0
    elif default_scenario == "utah" and spawnpoint == "undef":
        edges = bng.get_road_edges('15852')
        edges.extend(bng.get_road_edges('14904'))
        edges.extend(bng.get_road_edges('15316'))
        adjustment_factor = 1.0
    elif default_scenario == "driver_training" and spawnpoint == "approachingfork":
        edges = bng.get_road_edges("7719")
        edges.reverse()
        # edges = bng.get_road_edges('7936')
        # edges.extend(bng.get_road_edges('7836')) #7952
        adjustment_factor = 0.000001
    print("retrieved road edges")
    actual_middle = [edge['middle'] for edge in edges]
    print(f"{actual_middle[0]=}")
    roadleft = [edge['left'] for edge in edges]
    roadright = [edge['right'] for edge in edges]
    adjusted_middle = [np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle']))/adjustment_factor for edge in edges]
    centerline = actual_middle
    return actual_middle, adjusted_middle

def plot_trajectory(traj, title="Trajectory", label1="AI behavior"):
    global centerline, roadleft, roadright
    x = [t[0] for t in traj]
    y = [t[1] for t in traj]
    plt.plot(x,y, 'b', label=label1)
    # plt.gca().set_aspect('equal')
    # plt.axis('square')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'r-', label="centerline")
    plt.plot([t[0] for t in roadleft], [t[1] for t in roadleft], 'r-')
    plt.plot([t[0] for t in roadright], [t[1] for t in roadright], 'r-')
    plt.title(title)
    plt.legend()
    plt.draw()
    plt.savefig("{}/{}_collection_trajectory.jpg".format(training_dir, collection_hash))
    plt.show()
    plt.pause(0.1)

def intake_ai_lap_poses(filename="ai_lap_data.txt"):
    global centerline
    lap_traj = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        # lap_traj =
        for line in lines:
            line = line.replace("\n", "")
            # print(line)
            line = literal_eval(line)
            lap_traj.append(line)
    centerline = lap_traj
    return lap_traj

def get_start_index(adjusted_middle):
    global default_scenario, spawnpoint
    sp = spawn_point(default_scenario, spawnpoint)
    distance_from_centerline = dist_from_line(adjusted_middle, sp['pos'])
    # idx = adjusted_middle.index(min(distance_from_centerline))
    idx = max(np.where(distance_from_centerline == min(distance_from_centerline)))
    return idx[0]
    
def create_ai_line_from_road_with_interpolation(spawn, bng, swerving=False):
    global centerline, remaining_centerline, centerline_interpolated
    points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
    actual_middle, adjusted_middle = road_analysis(bng)
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
    for i,p in enumerate(actual_middle[:-1]):
        points.append([p[0], p[1], p[2]])
        point_colors.append([0, 1, 0, 0.1])
        spheres.append([p[0], p[1], p[2], 0.25])
        sphere_colors.append([1, 0, 0, 0.8])
    print("spawn point:{}".format(spawn))
    print("beginning of script:{}".format(middle[0]))
    plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
    centerline = copy.deepcopy(traj)
    remaining_centerline = copy.deepcopy(traj)
    centerline_interpolated = copy.deepcopy(traj)
    # for i in range(4):
    #     centerline.extend(copy.deepcopy(centerline))
    #     remaining_centerline.extend(copy.deepcopy(remaining_centerline))
    bng.add_debug_line(points, point_colors,
                       spheres=spheres, sphere_colors=sphere_colors,
                       cling=True, offset=0.1)
    return bng

def law_of_cosines(A, B, C):
    dist_AB = distance2D(A[:2], B[:2])
    dist_BC = distance2D(B[:2], C[:2])
    # p23 = distance2D(left, right)
    ab = np.array(B[:2]) - np.array(A[:2])
    bc = np.array(C[:2]) - np.array(B[:2])
    # return math.acos((math.pow(p12,2) + math.pow(p13,2) - math.pow(p23,2))/(2*p12*p13))
    return math.acos(np.dot(ab, bc) / (dist_AB * dist_BC))

def angle_between(next_waypoint, vehicle_state):
    # yaw = math.radians(vehicle_state['yaw'][0]) #
    # yaw = rpy_from_vec(vehicle_state['dir']); yaw = yaw[2]
    # reference_pt = [vehicle_state['pos'][0] + 5 * math.cos(yaw), vehicle_state['pos'][1] + 5 * math.sin(yaw)]
    # inner_angle = law_of_cosines(next_waypoint, vehicle_state['pos'], reference_pt)
    # waypoint_angle = math.tan(next_waypoint[1] / next_waypoint[0])
    # vehicle_angle = math.tan(vehicle_state['pos'][1] / vehicle_state['pos'][0])
    # inner_angle = waypoint_angle - vehicle_angle
    vehicle_angle = math.atan2(vehicle_state['front'][1]-vehicle_state['pos'][1], vehicle_state['front'][0]-vehicle_state['pos'][0])
    waypoint_angle = math.atan2((next_waypoint[1]-vehicle_state['pos'][1]),(next_waypoint[0]-vehicle_state['pos'][0]))
    inner_angle = -(waypoint_angle - vehicle_angle)
    # print("next_waypoint",next_waypoint)
    # print("vehicle_state", vehicle_state)
    # # print("vehicle_state['pos']", vehicle_state['pos'])
    # print("vehicle_angle (degrees):", math.degrees(vehicle_angle))
    # print("waypoint_angle (degrees):",math.degrees(waypoint_angle))
    # # print("inner_angle (degrees):",math.degrees(inner_angle))
    # print("inner_angle (degrees):", math.degrees(math.atan2(math.sin(inner_angle), math.cos(inner_angle))))
    return math.atan2(math.sin(inner_angle), math.cos(inner_angle))

def steering_setpoint(vehicle_state, traj=None):
    global remaining_centerline, centerline, throttle_setpoint, overall_throttle_setpoint
    global steer_integral, steer_prev_error, steer_prev_setpoint
    global tight_curve, avg_error
    if len(remaining_centerline) < 20:
        remaining_centerline.append(centerline_interpolated)
    # update next waypoint if you've reached your current one
    # normal driving circumstances
    # is there an upcoming tight curve?
    # shallow curve coming up soon
    if (angle_between(remaining_centerline[5], vehicle_state) > (math.pi / 6) or
        angle_between(remaining_centerline[9], vehicle_state) < -(math.pi / 8)): #6.55):
        slowdown = True
    else:
        slowdown = False
    if not tight_curve and (angle_between(remaining_centerline[3], vehicle_state) > (math.pi / 3.25) or
        angle_between(remaining_centerline[8], vehicle_state) < -(math.pi / 6.75)): #6.55):
        # print(abs(angle_between(remaining_centerline[0], vehicle_state)), "> (math.pi / 6)")
        tight_curve = True
    else:
        tight_curve = False

    # steep curve coming up in near future
    i = 0; a = 0
    while distance2D(vehicle_state['pos'][:2], remaining_centerline[i]) < 20: #15: #10: #13
        i += 1
        a = angle_between(remaining_centerline[i], vehicle_state)
        if abs(a) > (math.pi / 3.5):
            # print(abs(a), "> math.pi/4")
            tight_curve = True
            print("TIGHT CURVE!!!!!\nTIGHT CURVE!!!!!\nTIGHT CURVE!!!!!")
            break
    # print("lookahead angle:", a)
    changed = False
    if tight_curve or slowdown:
        throttle_setpoint = 32.5
        steer_integral = 0; steer_prev_error = 0
    else:
        if throttle_setpoint == 32.5:
            steer_integral = 0; steer_prev_error = 0
        i = 0
        throttle_setpoint = overall_throttle_setpoint
        while distance2D(vehicle_state['pos'][:2], remaining_centerline[i]) < 7.5:
            i += 1
            # changed = True
        if i > 0:
            steer_integral = 0; steer_prev_error = 0
    remaining_centerline = remaining_centerline[i:]
    next_waypoint = remaining_centerline[0]
    inner_angle = angle_between(next_waypoint, vehicle_state)
    # print("distance to next waypoint:",distance2D(vehicle_state['pos'][:2], next_waypoint[:2]))
    # plt.figure(figsize=(8,8))
    if changed:
        plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'r-', label="AI line script")
        plt.scatter(next_waypoint[0], next_waypoint[1], marker="o", label="next_waypoint")
        plt.scatter(vehicle_state['pos'][0], vehicle_state['pos'][1], marker="v", label="vehicle pos")
        plt.scatter(0, 0, marker="X", label="origin")
        plt.axis('square')
        plt.legend()
        plt.show()
        plt.pause(0.01)
    return math.atan2(math.sin(inner_angle), math.cos(inner_angle))

def plot_inputs(timestamps, inputs, title="Inputs by Time"):
    plt.plot(timestamps, inputs)
    plt.title(title)
    plt.show()
    plt.pause(0.01)

def dist_from_line(centerline, point):
    a = [[x[0],x[1]] for x in centerline[:-1]]
    b = [[x[0],x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist

# track is approximately 12.50m wide
# car is approximately 1.85m wide
def has_car_left_track(vehicle_pos, vehicle_bbox, bng):
    global centerline_interpolated
    # get nearest road point
    distance_from_centerline = dist_from_line(centerline_interpolated, vehicle_pos)
    # check if it went over left edge
    # print("Distance from center of road:", min(distance_from_centerline))
    return min(distance_from_centerline) > 15.0

def laps_completed(lapcount):
    global centerline_interpolated, remaining_centerline
    remainder = centerline_interpolated.index(remaining_centerline[0])
    remainder = remainder / len(centerline_interpolated)
    return lapcount + remainder

# kph setpoints, avg kph, # laps completed, just-recovery-behavior, swervingstdev, collectionscriptcommit
def save_collection_metas(overall_kph_setpt, curve_kph_setpt, avg_kph, outcome, total_laps, just_recovery, swervingstdev): #, scriptcommit):
    filename = "{}/{}-collection-metas.txt".format("/".join(training_dir.split("/")), collection_hash)
    with open(filename, "w") as f:
        f.write(f"{overall_kph_setpt=}\n"
                f"{curve_kph_setpt=}\n"
                f"{avg_kph=}\n"
                f"outcome={outcome.strip(', exiting...')}\n"
                f"{total_laps=}\n"
                f"{just_recovery=}\n"
                f"{swervingstdev=}\n")
    shutil.copyfile("{}/deepbillboard-collect-dataset.py".format(os.getcwd()), "{}/deepbillboard-collect-dataset-{}.py".format(training_dir, collection_hash))

def collection_run(speed=11, risk=0.6, num_samples=10000):
    global base_filename, training_dir, default_model, setpoint
    global spawnpoint, steps_per_second

    f = setup_dir(training_dir)
    spawn_pt = spawn_point(default_scenario, spawnpoint)
    # random.seed(1703)
    setup_logging()

    home = 'H:/BeamNG.research.v1.7.0.1clean' #'H:/BeamNG.tech.v0.21.3.0' #
    beamng = BeamNGpy('localhost', 64256, home=home, user='H:/BeamNG.research')
    scenario = Scenario(default_scenario, 'research_test')

    # add barriers and cars to get the ego vehicle to avoid the barriers
    add_barriers(scenario)
    vehicle = Vehicle('ego_vehicle', model=default_model, licence='EGO', color='White')
    vehicle = setup_sensors(vehicle)
    scenario.add_vehicle(vehicle, pos=spawn_pt['pos'], rot=None, rot_quat=spawn_pt['rot_quat'])

    # setup free camera
    # eagles_eye_cam = Camera((0, -350, 300.5),
    #                         (0.013892743289471, -0.015607489272952, -1.39813470840454, 0.91656774282455),
    #                         fov=90, resolution=(1500,1500),
    #                       colour=True, depth=True, annotation=True)
    # scenario.add_camera(eagles_eye_cam, "eagles_eye_cam")

    # Compile the scenario and place it in BeamNG's map folder
    scenario.make(beamng)
    bng = beamng.open(launch=True)
    # bng.hide_hud()
    bng.set_nondeterministic()
    bng.set_steps_per_second(steps_per_second)
    bng.load_scenario(scenario)
    bng.start_scenario()

    # collect overhead view of setup
    # freecams = scenario.render_cameras()
    # plt.title("freecam")
    # plt.imshow(freecams['eagles_eye_cam']["colour"].convert('RGB'))
    # freecams['eagles_eye_cam']["colour"].convert('RGB').save("eagles-eye-view.jpg", "JPEG")
    # plt.pause(0.01)

    plot_racetrack_roads(bng.get_roads(), bng)
    bng = create_ai_line_from_road_with_interpolation(spawn_pt, bng, swerving=False)

    bng.pause()
    assert vehicle.skt
    start_time = time.time()
    print(f"{start_time=}")
    # Send random inputs to vehicle and advance the simulation 20 steps
    imagecount = 0; lapcount =0
    timer = 0; traj = []; steering_inputs = []; timestamps = []; kphs = []
    steering=0; throttle=0; brake=0
    with open(f, 'w') as datafile:
        datafile.write('filename,timestamp,steering_input,throttle_input,brake_input,steering,engine_load,fuel,'
                       'lowpressure,parkingbrake,rpm,wheelspeed,'
                       'vel,pos,front,dir,up,gx_smooth_max,gx,gy,gz,gx2,gy2,gz2\n') # added 12 new vars
        return_str = ''
        print("file opened")
        while imagecount < num_samples:
            sensors = bng.poll_sensors(vehicle)
            # print(f"{sensors['electrics']['steering_input']=}")
            if sensors['electrics']['steering_input'] == 1:
                print(f"{sensors['electrics']['steering']=}")
            # print(f"kph={sensors['electrics']['wheelspeed'] * 3.6}")
            image = sensors['front_cam']['colour'].convert('RGB')
            full_filename = "{}{}{}.jpg".format(training_dir, base_filename, imagecount)
            qualified_filename = "{}{}.jpg".format(base_filename, imagecount)
            steering = sensors['electrics']['steering_input']
            dt = sensors['timer']['time'] - timer
            kph = sensors['electrics']['wheelspeed'] * 3.6

            # print("current kph:{} dt:{} throttle:{}".format(sensors['electrics']['wheelspeed'] * 3.6, dt, throttle))
            vehicle.update_vehicle()
            # try:
            steering_setpt = steering_setpoint(vehicle.state, traj)
            if dt > 0:
                steering = steering_PID(steering, steering_setpt, dt)
            # if throttle_setpoint <= 35 and kph > 30:
            #     throttle = 0
            #     brake = 0.125
            # else:
            #     brake = 0
            if dt > 0:
                throttle = throttle_PID(kph, dt)
            print(f"{steering=} {throttle=} {brake=}")
            vehicle.control(steering=steering, throttle=throttle, brake=brake)
            vehicle.update_vehicle()
            steering_input = sensors['electrics']['steering_input']
            if abs(steering_input) > 0.25 and kph > 30:
                throttle_setpoint = throttle_curve_setpt
                throttle = 0
                brake = 0.125
            else:
                throttle_setpoint = overall_throttle_setpoint
                brake = 0
            timer = sensors['timer']['time']
            traj.append(vehicle.state['pos'])
            timestamps.append(timer)
            steering_inputs.append(steering)
            kphs.append(kph)
            # except IndexError as e:
            #     plot_trajectory(traj, "Car Behavior using AI Script")
            #     plot_inputs(timestamps, steering_inputs, title="Steering Inputs by Time")
            #     plot_inputs(timestamps, kphs, title="KPHs by Time")
            # 'vel,pos,dir,front,up,gx_smooth_max,gx,gy,gz,gx2,gy2,gz2\n')  # added 12 new vars
            if timer > 10:
                 datafile.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                                                    qualified_filename,
                                                    str(round(sensors['timer']['time'], 2)),
                                                    sensors['electrics']['steering_input'],
                                                    sensors['electrics']['throttle_input'],
                                                    sensors['electrics']['brake_input'],
                                                    sensors['electrics']['steering'],
                                                    sensors['electrics']['engine_load'],
                                                    sensors['electrics']['fuel'],
                                                    sensors['electrics']['lowpressure'],
                                                    sensors['electrics']['parkingbrake'],
                                                    sensors['electrics']['rpm'],
                                                    sensors['electrics']['wheelspeed'],
                                                    str(vehicle.state['vel'][1:-1]).replace(",",""),
                                                    str(vehicle.state['pos'][1:-1]).replace(",", ""),
                                                    str(vehicle.state['front'][1:-1]).replace(",", ""),
                                                    str(vehicle.state['dir'][1:-1]).replace(",", ""),
                                                    str(vehicle.state['up'][1:-1]).replace(",", ""),
                                                    sensors['gforces']['gx_smooth_max'],
                                                    sensors['gforces']['gx'],
                                                    sensors['gforces']['gy'],
                                                    sensors['gforces']['gz'],
                                                    sensors['gforces']['gx2'],
                                                    sensors['gforces']['gy2'],
                                                    sensors['gforces']['gz2'])
                                                    )
                 # save the image
                 image.save(full_filename)
                 imagecount += 1
            # total_laps = laps_completed(lapcount)
            # print(f"{total_laps=}")
            print(f"{sensors['timer']['time']=}")
            if distance(spawn_pt['pos'], vehicle.state['pos']) < 8 and sensors['timer']['time'] > 180:
                # return_str = "Completed one lap, exiting..."
                # break
                lapcount += 1
            if sensors['damage']['damage'] > 0:
                return_str = "CRASHED at timestep {} speed {}; QUITTING".format(round(sensors['timer']['time'], 2), round(sensors['electrics']['wheelspeed']*3.6, 3))
                print(return_str)
                break
            outside_track = has_car_left_track(vehicle.state['pos'], vehicle.get_bbox(), bng)
            if outside_track:
                return_str = "Left track, exiting..."
                break
            bng.step(1, wait=True)

    total_laps = laps_completed(lapcount)
    print("RESULTS OF DATASET COLLECTION")
    print(f"\t{imagecount=}")
    print(f"\t{total_laps=}")
    print(f"\t{return_str}")
    print(f"\tFiles written to {training_dir}")
    plot_trajectory(traj, "Car Behavior using AI Script")
    # plot_inputs(timestamps, steering_inputs, title="Steering Inputs by Time")
    # plot_inputs(timestamps, kphs, title="KPHs by Time")
    bng.close()
    save_collection_metas(overall_throttle_setpoint, throttle_curve_setpt, sum(kphs) / len(kphs), return_str, total_laps,
                          swerving, swervingstdev)
    # plot_trajectory(traj, "Car Behavior using AI Script")
    return return_str

def main():
    global numsamps
    collection_run(speed=12, risk=0.3, num_samples=numsamps)

if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').disabled = True
    main()
