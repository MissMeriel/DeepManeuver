"""
.. module:: collect_images
    :platform: Windows
    :synopsis: Collect samples of runs and their outcomes

.. moduleauthor:: Meriel von Stein <meriel@virginia.edu>

"""

import numpy as np
from matplotlib import pyplot as plt
import random, copy

from beamngpy import StaticObject, ScenarioObject, ProceduralCube
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy import ProceduralCube #,ProceduralCylinder, ProceduralCone, ProceduralBump, ProceduralRing
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

import math
from scipy.spatial.transform import Rotation as R
from ast import literal_eval
from scipy import interpolate
import statistics

# MAKE SURE TO PULL METHODS FROM COLLECT_DATASET
class Simulator():

    def __init__(self, scenario_name="industrial", spawnpoint_name='racetrackstartinggate',
                 vehicle_model='hopper', steps_per_sec=15,
                 lap_filename="posefiles/DAVE2v3-lap-trajectory.txt",
                 path2sim=""):
        self.scenario_name = scenario_name
        self.spawnpoint_name = spawnpoint_name
        self.remaining_centerline = []
        self.centerline = []
        self.centerline_interpolated = []
        self.roadleft, self.roadright, self.roadmiddle = [], [], []
        self.landwidth = 3.75
        self.roadwidth = 9.0
        self.steps_per_sec = steps_per_sec
        self.qr_positions = []
        self.expected_trajectory = self.intake_lap_file(lap_filename)
        self.beamng = BeamNGpy('localhost', 64256, home=f'{path2sim}/BeamNG.research.v1.7.0.1', user=f'{path2sim}/BeamNG.research')
        self.scenario = Scenario(self.scenario_name, 'research_test')
        self.vehicle = Vehicle('ego_vehicle', model=vehicle_model, licence='EGO', color='White')
        self.setup_sensors()
        sp = self.spawn_point(self.scenario_name, self.spawnpoint_name)
        self.scenario.add_vehicle(self.vehicle, pos=sp['pos'], rot=None, rot_quat=sp['rot_quat'])
        self.add_barriers()
        self.add_qr_cubes()
        self.scenario.make(self.beamng)
        self.bng = self.beamng.open(launch=True)
        self.bng.set_steps_per_second(self.steps_per_sec)
        self.bng.set_deterministic()
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        self.bng.pause()
        assert self.vehicle.skt
        self.create_ai_line_from_road_with_interpolation(sp)

    def get_sequence_setup(self, sequence):
        import pandas as pd
        df = pd.read_csv("posefiles/sequence-setup.txt", sep="\s+")
        keys = df.keys()
        index = df.index[df['sequence'] == sequence].tolist()[0]
        vals = {key: df.at[index, key] for key in keys}
        return vals

    def spawn_point(self, scenario_locale, spawn_point='default'):
        if scenario_locale == 'industrial':
            if spawn_point == "curve1":
                # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.002,0.004,0.920,-0.391)}
                return {'pos': (210.314, -44.7132, 42.7758), 'rot': None, 'rot_quat': (0.002, 0.005, 0.920, -0.391)}
            elif spawn_point == "straight1":
                # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.002,0.004,0.920,-0.391)}
                # return {'pos': (252.028,-24.7376,42.814), 'rot': None,'rot_quat': (-0.044,0.057,-0.496,0.866)} # 130 steps
                return {'pos': (257.414, -27.9716, 42.8266), 'rot': None, 'rot_quat': (-0.0324, 0.0535, -0.451, 0.890)} # 50 steps
                return {'pos': (265.087, -33.7904, 42.805), 'rot': None, 'rot_quat': (-0.0227, 0.0231, -0.4228, 0.9056)} # 4 steps
            elif spawn_point == "curve2":
                # return {'pos':(323.432,-92.7588,43.6475), 'rot': None, 'rot_quat':(0.008,0.0138,-0.365,0.931)}
                return {'pos': (331.169, -104.166, 44.142), 'rot': None, 'rot_quat': (0.010, 0.0337, -0.359, 0.933)}

    def turn_X_degrees(self, rot_quat, degrees=90):
        r = R.from_quat(list(rot_quat))
        r = r.as_euler('xyz', degrees=True)
        r[2] = r[2] + degrees
        r = R.from_euler('xyz', r, degrees=True)
        return tuple(r.as_quat())

    def calc_deviation_from_center(self, traj, baseline=None):
        # dists = np.zeros(len(traj))
        # for i, point in enumerate(traj):
        #     dists[i] = dist_from_line(centerline, point)
        # avg_dist = sum(dists) / len(dists)
        # stddev = statistics.stdev(dists)
        # return stddev, dists, avg_dist
        dists = []
        for point in traj:
            if baseline is not None:
                dist = self.dist_from_line(baseline, point)
            else:
                dist = self.dist_from_line(self.expected_trajectory, point)
            dists.append(min(dist))
        avg_dist = sum(dists) / len(dists)
        stddev = statistics.stdev(dists)
        return stddev, dists, avg_dist

    def nearest_seg(self):
        road_seg = {}
        dists = self.dist_from_line(self.roadmiddle, self.vehicle.state['front'])
        idx = max(np.where(dists == min(dists)))[0]
        road_seg_left = []
        road_seg_right = []
        # road_seg_center = []
        for i in range(-1, 15):
            if idx + i < 0:
                road_seg_left.append(self.roadleft[len(self.roadleft) + (idx + i)])
                road_seg_right.append(self.roadright[len(self.roadright) + (idx + i)])
            else:
                road_seg_left.append(self.roadleft[idx + i])
                road_seg_right.append(self.roadright[idx + i])
        road_seg['left'] = road_seg_left
        road_seg['right'] = road_seg_right
        return road_seg

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

    def rpy_from_vec(self, vec):
        r = R.from_rotvec(list(vec))
        r = r.as_euler('xyz', degrees=True)
        return r

    def distance(self, a, b):
        if len(a) == 2 or len(b) == 2:
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        else:
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def add_barriers(self, filename='posefiles/industrial_racetrack_barrier_locations.txt'):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(' ')
                pos = line[0].split(',')
                pos = tuple([float(i) for i in pos])
                rot_quat = line[1].split(',')
                rot_quat = tuple([float(j) for j in rot_quat])
                rot_quat = self.turn_X_degrees(rot_quat, 90)
                ramp = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
                                    shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
                self.scenario.add_object(ramp)

    def add_qr_cubes(self, filename='posefiles/qr_box_locations.txt'):
        self.qr_positions = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(' ')
                pos = line[0].split(',')
                pos = tuple([float(i) for i in pos])
                rot_quat = line[1].split(',')
                rot_quat = tuple([float(j) for j in rot_quat])
                self.qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
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
                self.scenario.add_object(box)
            cube = ProceduralCube(name='cube_platform',
                                  pos=(145.214,-160.72,43.7269), rot=None, rot_quat=(0, 0, 0, 1),
                                  size=(2, 6, 0.5))
            self.scenario.add_procedural_mesh(cube)

    def from_val_get_key(self, d, v):
        key_list = list(d.keys())
        val_list = list(d.values())
        position = val_list.index(v)
        return key_list[position]

    def plot_trajectory(self, traj, title="Trajectory", label1="AI behavior"):
        # global centerline, roadleft, roadright, new_results_dir, default_scenario, default_spawnpoint, qr_positions
        x = [t[0] for t in traj]
        y = [t[1] for t in traj]
        plt.plot(x, y, 'b', label=label1)
        # plt.gca().set_aspect('equal')
        # plt.axis('square')
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.plot([t[0] for t in centerline], [t[1] for t in centerline], 'k-', label="centerline")
        plt.plot([t[0] for t in roadleft], [t[1] for t in roadleft], 'r-', label="left")
        plt.plot([t[0] for t in roadright], [t[1] for t in roadright], 'g-', label="right")
        sp = self.spawn_point(self.scenario_name, self.spawnpoint_name)
        plt.scatter(sp['pos'][0], sp['pos'][1], marker="o", linewidths=10, label="spawnpoint")
        plt.plot([p[0][0] for p in self.qr_positions], [p[0][1] for p in self.qr_positions], linewidth=5, label="billboard")
        plt.title(title)
        plt.legend()
        plt.draw()
        if new_results_dir == '':
            plt.savefig("{}/{}-{}_expected-trajectory.jpg".format(os.getcwd(), self.scenario_name, self.spawnpoint_name))
        else:
            plt.savefig(
                "{}/{}-{}_expected-trajectory.jpg".format(new_results_dir, self.scenario_name, self.spawnpoint_name))
        # plt.show()
        # plt.pause(1)
        plt.close("all")

    def plot_racetrack_roads(self, bng):
        roads = self.bng.get_roads()
        colors = ['b','g','r','c','m','y','k']
        symbs = ['-','--','-.',':','.',',','v','o','1',]
        print(f"{len(roads)=}")
        for road in roads:
            road_edges = bng.get_road_edges(road)
            x_temp, y_temp = [], []
            xy_def = [edge['middle'][:2] for edge in road_edges]
            dists = [self.distance(xy_def[i][:2], xy_def[i+1][:2]) for i,p in enumerate(xy_def[:-1])]
            if sum(dists) > 500:
                continue
            for edge in road_edges:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
        sp = self.spawn_point(self.scenario_name, self.spawnpoint_name)
        plt.scatter(sp['pos'][0], sp['pos'][1], marker="o", linewidths=10, label="spawnpoint")
        plt.legend()
        plt.title("{} {}".format(self.scenario_name, self.spawnpoint_name))
        plt.show()
        plt.pause(0.001)

    def road_analysis(self):
        # global centerline, roadleft, roadright
        # self.plot_racetrack_roads(self.bng)
        edges = []
        if self.scenario_name == "industrial" and self.spawnpoint_name == "racetrackstartinggate":
            edges = self.bng.get_road_edges('7983')
        elif self.scenario_name == "industrial" and self.spawnpoint_name == "driftcourse2":
            edges = self.bng.get_road_edges('7987')
        elif self.scenario_name == "industrial":
            edges = self.bng.get_road_edges('7982')
        elif self.scenario_name == "hirochi_raceway" and self.spawnpoint_name == "startingline":
            edges = self.bng.get_road_edges('9096')
            edges.extend(self.bng.get_road_edges('9206'))
            # edges = bng.get_road_edges('9206')
        elif self.scenario_name == "utah" and self.spawnpoint_name == "westhighway":
            edges = self.bng.get_road_edges('15145')
            # edges.extend(bng.get_road_edges('15162'))
            edges.extend(self.bng.get_road_edges('15154'))
            edges.extend(self.bng.get_road_edges('15810'))
            edges.extend(self.bng.get_road_edges('16513'))
        elif self.scenario_name == "utah" and self.spawnpoint_name == "westhighway2":
            edges = self.bng.get_road_edges('15810')
            # edges.extend(bng.get_road_edges('15810'))
            edges.extend(self.bng.get_road_edges('16513'))
            # edges.extend(bng.get_road_edges('15143'))
            # edges = bng.get_road_edges('9206')
        elif self.scenario_name == "utah" and self.spawnpoint_name == "undef":
            edges = self.bng.get_road_edges('15852')
            edges.extend(self.bng.get_road_edges('14904'))
            edges.extend(self.bng.get_road_edges('15316'))
        elif self.scenario_name == "driver_training" and self.spawnpoint_name == "approachingfork":
            edges = self.bng.get_road_edges("7719")
            edges.reverse()
            # edges = bng.get_road_edges('7936')
            # edges.extend(bng.get_road_edges('7836')) #7952
        actual_middle = [edge['middle'] for edge in edges]
        self.roadleft = [edge['left'] for edge in edges]
        self.roadright = [edge['right'] for edge in edges]
        adjusted_middle = [np.array(edge['middle']) + (np.array(edge['left']) - np.array(edge['middle']))/4.0 for edge in edges]
        self.centerline = actual_middle
        return actual_middle, adjusted_middle

    def get_start_index(self, adjusted_middle):
        sp = self.spawn_point(self.scenario_name, self.spawnpoint_name)
        distance_from_centerline = self.dist_from_line(adjusted_middle, sp['pos'])
        idx = max(np.where(distance_from_centerline == min(distance_from_centerline)))
        return idx[0]

    def create_ai_line_from_road_with_interpolation(self, spawn):
        # global centerline, remaining_centerline, centerline_interpolated
        points = []; point_colors = []; spheres = []; sphere_colors = []; traj = []
        actual_middle, adjusted_middle = self.road_analysis()
        middle_end = adjusted_middle[:3]
        middle = adjusted_middle[3:]
        middle.extend(middle_end)
        middle.append(middle[0])
        self.roadmiddle = middle
        # middle = list(np.roll(np.array(adjusted_middle), 3*(len(adjusted_middle) - 4)))
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
        # self.plot_trajectory(traj, "Points on Script (Final)", "AI debug line")
        self.centerline = copy.deepcopy(traj)
        self.remaining_centerline = copy.deepcopy(traj)
        self.centerline_interpolated = copy.deepcopy(traj)
        # for i in range(4):
        #     centerline.extend(copy.deepcopy(centerline))
        #     remaining_centerline.extend(copy.deepcopy(remaining_centerline))
        self.bng.add_debug_line(points, point_colors,
                           spheres=spheres, sphere_colors=sphere_colors,
                           cling=True, offset=0.1)
        # return (points, point_colors, spheres, sphere_colors, True, 0.1)

    def find_width_of_road(self, road_id='7983'):
        edges = self.bng.get_road_edges(road_id)
        left_edge = [edge['left'] for edge in edges]
        right_edge = [edge['right'] for edge in edges]
        middle = [edge['middle'] for edge in edges]
        return self.distance(left_edge[0], middle[0]) + self.distance(right_edge[0], middle[0])

    def returned_to_expected_traj(self, pos_window):
        dists = []
        for point in pos_window:
            dist = self.dist_from_line(self.expected_trajectory, point)
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

    def intake_lap_file(self, filename="posefiles/DAVE2v1-lap-trajectory.txt"):
        expected_trajectory = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                line = literal_eval(line)
                expected_trajectory.append(line)
        return expected_trajectory

    # track ~12.50m wide, car ~1.85m wide
    def has_car_left_track(self): #, vehicle_pos, vehicle_bbox, bng):
        distance_from_centerline = self.dist_from_line(self.centerline_interpolated, self.vehicle.state['pos'])
        return min(distance_from_centerline) > 15.0


    def laps_completed(self, lapcount):
        remainder = self.centerline_interpolated.index(self.remaining_centerline[0])
        remainder = remainder / len(self.centerline_interpolated)
        return lapcount + remainder


    def adjust_centerline_for_spawn(self):
        return

    def setup_sensors(self, pos=(-0.5, 0.38, 1.3), direction=(0, 1.0, 0)):
        fov = 50  # 60 works for full lap #63 breaks on hairpin turn
        resolution = (240, 135)  # (400,225) # (240, 135)  # (200, 150) (320, 180) #(1280,960) #(512, 512)
        front_camera = Camera(pos, direction, fov, resolution,
                              colour=True, depth=True, annotation=True)
        gforces = GForces()
        electrics = Electrics()
        damage = Damage()
        timer = Timer()

        # Attach them
        self.vehicle.attach_sensor('front_cam', front_camera)
        self.vehicle.attach_sensor('gforces', gforces)
        self.vehicle.attach_sensor('electrics', electrics)
        self.vehicle.attach_sensor('damage', damage)
        self.vehicle.attach_sensor('timer', timer)

    def get_sensor_readings(self):
        self.vehicle.update_vehicle()
        return self.bng.poll_sensors(self.vehicle)

    def get_vehicle_state(self):
        self.vehicle.update_vehicle()
        return self.vehicle.state

    def restart(self):
        self.bng.restart_scenario()
        self.bng.pause()

    def drive(self, throttle=1.0, steering=0.0, brake=0.0):
        self.vehicle.control(throttle=throttle, steering=steering, brake=brake)

#######################################################################################################################
    # def add_barriers(scenario):
    # with open('posefiles/industrial_racetrack_barrier_locations.txt', 'r') as f:
    #         lines = f.readlines()
    #         for i, line in enumerate(lines):
    #             line = line.split(' ')
    #             pos = line[0].split(',')
    #             pos = tuple([float(i) for i in pos])
    #             rot_quat = line[1].split(',')
    #             rot_quat = tuple([float(j) for j in rot_quat])
    #             # turn barrier 90 degrees
    #             r = R.from_quat(list(rot_quat))
    #             r = r.as_euler('xyz', degrees=True)
    #             r[2] = r[2] + 90
    #             r = R.from_euler('xyz', r, degrees=True)
    #             rot_quat = tuple(r.as_quat())
    #             barrier = StaticObject(name='barrier{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(1, 1, 1),
    #                                 shape='levels/Industrial/art/shapes/misc/concrete_road_barrier_a.dae')
    #             # barrier.type="BeamNGVehicle"
    #             scenario.add_object(barrier)
    #
    # def add_qr_cubes(scenario):
    #     global qr_positions
    #     qr_positions = []
    #     with open('posefiles/qr_box_locations.txt', 'r') as f:
    #         lines = f.readlines()
    #         for i, line in enumerate(lines):
    #             line = line.split(' ')
    #             pos = line[0].split(',')
    #             pos = tuple([float(i) for i in pos])
    #             rot_quat = line[1].split(',')
    #             rot_quat = tuple([float(j) for j in rot_quat])
    #             qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
    #             # box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
    #             #                      rot_quat=rot_quat, scale=(10, 1, 5), JBeam='qrbox2', datablock="default_vehicle")
    #             # scale=(width, depth, height)
    #             # box = StaticObject(name='qrbox_{}'.format(i), pos=pos, rot=None, rot_quat=rot_quat, scale=(3, 0.1, 3),
    #             #                     shape='vehicles/metal_box/metal_box.dae')
    #             box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
    #                                  rot_quat=rot_quat, scale=(5,5,5), JBeam='qrbox2', datablock="default_vehicle")
    #             # cube = ProceduralCube(name='cube',
    #             #                       pos=pos,
    #             #                       rot=None,
    #             #                       rot_quat=rot_quat,
    #             #                       size=(5, 2, 3))
    #             # cube.type = 'BeamNGVehicle'
    #             # scenario.add_procedural_mesh(cube)
    #             scenario.add_object(box)
    #         if default_scenario == "industrial" and default_spawnpoint == "curve1":
    #             cube = ProceduralCube(name='cube_platform',
    #                                   pos=(145.214,-160.72,43.7269),
    #                                   rot=None,
    #                                   rot_quat=(0, 0, 0, 1),
    #                                   size=(2, 6, 0.5))
    #             scenario.add_procedural_mesh(cube)
    #         elif default_scenario == "driver_training" and default_spawnpoint == "approachingfork":
    #             cube = ProceduralCube(name='cube_platform',
    #                                   pos=(-20.3113, 218.448, 50.043),
    #                                   rot=None,
    #                                   rot_quat=(-0.022064134478569,-0.022462423890829,0.82797580957413,0.55987912416458),
    #                                   size=(4, 8, 0.5))
    #             scenario.add_procedural_mesh(cube)


    def setup_in_opposite_direction(self):
        return

    # uses blob detection
    # def get_qr_corners_from_colorseg_image_nowarp(image):
    #     image = np.array(image)
    #     orig_image = copy.deepcopy(image)
    #
    #     # mask + convert image to inverted greyscale
    #     hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     light_color = (50, 230, 0)  # (50, 235, 235) #(0, 200, 0)
    #     dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
    #     mask = cv2.inRange(hsv_image, light_color, dark_color)
    #     image = cv2.bitwise_and(image, image, mask=mask)
    #     R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    #     imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    #     inverted_img = util.invert(imgGray)
    #     inverted_img = np.uint8(inverted_img)
    #     inverted_img = 255 - inverted_img
    #
    #     detector = cv2.SimpleBlobDetector_create()
    #     keypoints = detector.detect(inverted_img)
    #     if keypoints == []:
    #         # print("No QR code detected")
    #         return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
    #     else:
    #         # ORDER: upper left, upper right, lower left, lower right
    #         bboxes = [[(int(keypoint.pt[0] - keypoint.size / 2), int(keypoint.pt[1] - keypoint.size / 2)),
    #                    (int(keypoint.pt[0] + keypoint.size / 2), int(keypoint.pt[1] - keypoint.size / 2)),
    #                    (int(keypoint.pt[0] - keypoint.size / 2), int(keypoint.pt[1] + keypoint.size / 2)),
    #                    (int(keypoint.pt[0] + keypoint.size / 2), int(keypoint.pt[1] + keypoint.size / 2))] for keypoint
    #                   in
    #                   keypoints]
    #         boxedimg = cv2.rectangle(orig_image, bboxes[0][0], bboxes[0][3], (255, 0, 0), 1)
    #         cv2.imshow('boxedimg', boxedimg)
    #         cv2.waitKey(1)
    #         return bboxes, boxedimg

    # def add_perturbed_billboard(img, bb, qr_corners):
    #     # size = (qr_corners[3][0] - qr_corners[0][0], qr_corners[3][1] - qr_corners[0][1])
    #     # resized_bb = cv2.resize(bb, size)
    #     # img = overlay_transparent_nowarp(np.array(img), np.array(resized_bb), qr_corners[0][0], qr_corners[0][1])
    #     img = overlay_transparent(np.array(img), bb, np.asarray(qr_corners))
    #     return img

