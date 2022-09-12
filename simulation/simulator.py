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
                 bb_filename="posefiles/DAVE2v3-lap-trajectory.txt",
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
        self.add_qr_cubes(bb_filename)
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

    def spawn_point(self, scenario_locale, road_id='default'):
        if scenario_locale == 'industrial':
            if road_id == "curve1":
                # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.002,0.004,0.920,-0.391)}
                return {'pos': (210.314, -44.7132, 42.7758), 'rot': None, 'rot_quat': (0.002, 0.005, 0.920, -0.391)}
            elif road_id == "straight1":
                # return {'pos':(189.603,-69.0035,42.7829), 'rot': None, 'rot_quat':(0.002,0.004,0.920,-0.391)}
                # return {'pos': (252.028,-24.7376,42.814), 'rot': None,'rot_quat': (-0.044,0.057,-0.496,0.866)} # 130 steps
                return {'pos': (257.414, -27.9716, 42.8266), 'rot': None, 'rot_quat': (-0.0324, 0.0535, -0.451, 0.890)} # 50 steps
                return {'pos': (265.087, -33.7904, 42.805), 'rot': None, 'rot_quat': (-0.0227, 0.0231, -0.4228, 0.9056)} # 4 steps
            elif road_id == "curve2":
                # return {'pos':(323.432,-92.7588,43.6475), 'rot': None, 'rot_quat':(0.008,0.0138,-0.365,0.931)}
                return {'pos': (331.169, -104.166, 44.142), 'rot': None, 'rot_quat': (0.010, 0.0337, -0.359, 0.933)}
        elif scenario_locale == 'west_coast_usa':
            if road_id == 'midhighway':
                # mid highway scenario (past shadowy parts of road)
                return {'pos': (-145.775, 211.862, 115.55), 'rot': None, 'rot_quat': (0.0032586499582976, -0.0018308814615011, 0.92652350664139, -0.37621837854385)}
            # actually past shadowy parts of road?
            #return {'pos': (95.1332, 409.858, 117.435), 'rot': None, 'rot_quat': (0.0077012465335429, 0.0036200874019414, 0.90092438459396, -0.43389266729355)}
            # surface road (crashes early af)
            elif road_id == '12669':
                return {'pos': (456.85526276, -183.39646912,  145.54124832), 'rot': None, 'rot_quat': turn_X_degrees((-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922), 90)}
            elif road_id == 'surfaceroad1':
                return {'pos': (945.285, 886.716, 132.061), 'rot': None, 'rot_quat': (-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922)}
            # surface road 2
            elif road_id == 'surfaceroad2':
                return {'pos': (900.016, 959.477, 127.227), 'rot': None, 'rot_quat': (-0.046136282384396, 0.018260028213263, 0.94000166654587, 0.3375423848629)}
            # surface road 3 (start at top of hill)
            elif road_id == 'surfaceroad3':
                return {'pos':(873.494, 984.636, 125.398), 'rot': None, 'rot_quat':(-0.043183419853449, 2.3034785044729e-05, 0.86842048168182, 0.4939444065094)}
            # surface road 4 (right turn onto surface road) (HAS ACCOMPANYING AI DIRECTION AS ORACLE)
            elif road_id == 'surfaceroad4':
                return {'pos': (956.013, 838.735, 134.014), 'rot': None, 'rot_quat': (0.020984912291169, 0.037122081965208, -0.31912142038345, 0.94675397872925)}
            # surface road 5 (ramp past shady el)
            elif road_id == 'surfaceroad5':
                return {'pos':(166.287, 812.774, 102.328), 'rot': None, 'rot_quat':(0.0038638345431536, -0.00049926445353776, 0.60924011468887, 0.79297626018524)}
            # entry ramp going opposite way
            elif road_id == 'entryrampopp':
                return {'pos': (850.136, 946.166, 123.827), 'rot': None, 'rot_quat': (-0.030755277723074, 0.016458060592413, 0.37487033009529, 0.92642092704773)}
            # racetrack
            elif road_id == 'racetrack':
                return {'pos': (395.125, -247.713, 145.67), 'rot': None, 'rot_quat': (0, 0, 0.700608, 0.713546)}
        elif scenario_locale == 'smallgrid':
            return {'pos':(0.0, 0.0, 0.0), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
            # right after toll
            return {'pos': (-852.024, -517.391 + lanewidth, 106.620), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}
            # return {'pos':(-717.121, 101, 118.675), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
            return {'pos': (-717.121, 101, 118.675), 'rot': None, 'rot_quat': (0, 0, 0.918812, -0.394696)}
        elif scenario_locale == 'automation_test_track':
            if road_id == '8127': # dirt road
                return {'pos': (121.2, -314.8, 123.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.2777, 0.9607), 20)}
            elif road_id == "8304": # big bridge/tunnel road, concrete walls
                # return {'pos': (357.2, 741.5, 132.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 60)}
                return {'pos': (324.593,655.58,132.642), 'rot': None, 'rot_quat': (-0.007, 0.005, 0.111, 0.994)}
            elif road_id == "8301": # country road next to river
                return {'pos': (262.0, -289.3, 121), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -60)}
            elif road_id == "8394": # long straight tree-lined country road near gas stn.
                return {'pos': (-582.0, -249.2, 117.3), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -60)}
            elif road_id == "7991": # long road, spawns atop overpass bridge
                return {'pos': (57.229, 360.560, 128.203), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
            elif road_id == "8293": # winding mountain road, lanelines
                return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.702719, 0.711467), 120)}
            elif road_id == "8205": # starting line, same as default
                # return {'pos': (501.36,178.62,131.69), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 0)} # orig
                return {'pos': (517.08, 178.84, 132.2), 'rot': None, 'rot_quat': (-0.0076747848652303, -0.0023407069966197, -0.70286595821381, 0.71127712726593)}  # closer
            elif road_id == "8185":  # bridge, lanelines
                return {'pos': (174.92, -289.67, 120.67), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
            else: # default
                return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif scenario_locale == 'industrial':
            if road_id == 'west':
                # western industrial area -- didnt work with AI Driver
                return {'pos': (237.131, -379.919, 34.5561), 'rot': None, 'rot_quat': (-0.035, -0.0181, 0.949, 0.314)}
            # open industrial area -- didnt work with AI Driver
            # drift course (dirt and paved)
            elif road_id == 'driftcourse':
                return {'pos': (20.572, 161.438, 44.2149), 'rot': None, 'rot_quat': (-0.003, -0.005, -0.636, 0.771)}
            # rallycross course/default
            elif road_id == 'rallycross':
                return {'pos': (4.85287, 160.992, 44.2151), 'rot': None, 'rot_quat': (-0.0032, 0.003, 0.763, 0.646)}
            # racetrack
            elif road_id == 'racetrackright':
                return {'pos': (184.983, -41.0821, 42.7761), 'rot': None, 'rot_quat': (-0.005, 0.001, 0.299, 0.954)}
            elif road_id == 'racetrackleft':
                return {'pos': (216.578, -28.1725, 42.7788), 'rot': None, 'rot_quat': (-0.0051, -0.003147, -0.67135, 0.74112)}
            elif road_id == 'racetrackstartinggate':
                return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036, 0.0065, 0.9234, -0.3837)}
            elif road_id == "racetrackstraightaway":
                return {'pos':(262.328, -35.933, 42.5965), 'rot': None, 'rot_quat':(-0.0105, 0.02997, -0.4481, 0.8934)}
            elif road_id == "racetrackcurves":
                return {'pos':(215.912,-243.067,45.8604), 'rot': None, 'rot_quat':(0.0290,0.0222,0.9860,0.1626)}
        elif scenario_locale == "hirochi_raceway":
            if road_id == "9039": # good candidate for input rect.
                # return {'pos': (290.558, -277.280, 46.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), -130)} #start
                # return {'pos': (490.073, -154.12, 35.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), -90)}  # 2nd quarter
                # centerline[ 614 ]: [557.3650512695312, -90.57571411132812, 43.21120071411133]
                # return {'pos': (557.4, -90.6, 43.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 35)}  # 2nd quarter
                # centerline[ 761 ]: [9.719991683959961, 342.0410461425781, 31.564104080200195]
                # return {'pos': (9.72, 342.0, 31.75), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 130)}  # 3rd quarter
                #centerline[ 1842 ]: [220.56719970703125, 365.5675048828125, 35.992027282714844]
                # return {'pos': (220.6, 365.6, 36.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 130)}  # 3rd quarter
                # centerline[1900]: [-32.84414291381836, 386.8899841308594, 36.25067901611328]
                # return {'pos': (-32.8, 386.9, 36.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 15)}  # 3rd quarter
                # centerline[1909]: [-45.08585739135742, 414.32073974609375, 38.64292526245117]
                # return {'pos': (-45.1, 414.3, 38.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 0)}  # 3rd quarter
                # centerline[2000]: [523.5741577148438, -135.5963134765625, 38.25138473510742]
                return {'pos': (523.6, -135.6, 38.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 90)}  # 3rd quarter
                # # centerline[4479]: [-346.49896240234375, 431.0029296875, 30.750564575195312]
                # return {'pos': (-346.5, 431.0, 30.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.960669), 0)}  # 3rd quarter
            elif road_id == "9205":
                return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            elif road_id == "9119":
                # return {"pos": (-452.972, 16.0, 29.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 200)} # before tollbooth
                # return {"pos": (-452.972, 64.0, 30.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 200)} # past tollbooth
                # centerline[ 150 ]: [-548.0482177734375, 485.4112243652344, 32.8107795715332]
                return {"pos": (-548.0, 485.4, 33.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -100)}
                # centerline[ 300 ]: [-255.45079040527344, 454.82879638671875, 28.71044921875]
                return {"pos": (-255.5, 454.8, 28.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
                # centerline[ 500 ]: [-317.8174743652344, 459.4542236328125, 31.227020263671875]
                return {"pos": (-317.8, 459.5, 31.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -50)}
                # centerline[ 1000 ]: [-421.6916809082031, 508.9856872558594, 36.54324722290039]
                return {"pos": (-421.7, 509, 36.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
                # centerline[ 1647 ]: [-236.64036560058594, 428.26605224609375, 29.6795597076416]
                return {"pos": (-236.6, 428.3, 29.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "9167":
                # veers away onto dirt road
                # return {'pos': (105.3, -96.4, 25.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 40)}
                # centerline[ 150 ]: [187.20333862304688, -146.42086791992188, 25.565567016601562]
                # return {'pos': (187.2, -146.4, 25.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 160)}
                # centerline[ 300 ]: [152.4763641357422, -257.7218933105469, 29.21633529663086]
                return {'pos': (152.5, -257.7, 29.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
                # centerline[ 1983 ]: [279.4185791015625, -261.2400817871094, 47.39253234863281]
                return {'pos': (279.4, -261.2, 47.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "9156":  # starting line
                # return {'pos': (-376.25, 200.8, 25.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 45)} # crashes into siderail
                # return {'pos': (-374, 90.2, 28.75), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 15)}  # past siderail
                return {'pos': (-301.314,-28.4299,32.9049), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 55)}  # past siderail
                # return {'pos': (-379.184,208.735,25.4121), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
                return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            elif road_id == "9189":
                # return {'pos': (-383.498, 436.979, 32.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
                return {'pos': (-447.6, 468.22, 32.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
            elif road_id == "9202": # lanelines
                # return {'pos': (-315.2, 80.94, 32.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -20)}
                # centerline[150]: [233.89662170410156, 88.48623657226562, 25.24519157409668]
                # return {'pos': (233.9, 88.5, 25.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -45)}
                # centerline[300]: [-244.8330078125, -9.06863784790039, 25.405052185058594]
                return {'pos': (-244.8, -9.1, 25.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -45)}
                # centerline[ 4239 ]: [-9.485580444335938, 17.10186004638672, 25.56268310546875]
            elif road_id == "9062":
                # return {'pos': (24.32, 231, 26.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -45)} # start
                # return {'pos': (155.3, 119.1, 25.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 70)} # middle
                return {'pos': (-82.7, 10.7, 25.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 110)} # end
            elif road_id == "9431": # actually a road edge
                return {'pos': (204.34,-164.94,25.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "9033": # road around a tan parking lot
                return {'pos': (-293.84,225.67,25.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
            elif road_id == "9055": # roadside edge
                return {'pos': (469.74,122.98,27.45), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "9064": # narrow, good for input rect, similar to orig track
                return {'pos': (-117.6, 201.2, 25.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 110)}
                return {'pos': (-93.30,208.40,25.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 110)} # start, crashes immediately
            elif road_id == "9069": # narrow, past dirt patch, lots of deviations
                return {'pos': (-77.27,-135.96,29.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "9095": # good for input rect, well formed, long runtime w/o crash
                return {'pos': (-150.15,174.55,32.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 180)}
            elif road_id == "9129": # good for input rect, long runtime w/o crash
                return {'pos': (410.96,-85.45,30.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
            elif road_id == "9189": # narrow, edged by fence, early crash
                return {'pos': (-383.50,436.98,32.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "9204": # lanelines, near tan parking lot as 9033, turns into roadside edge
                return {'pos': (-279.37,155.85,30.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -110)}
            else:
                return {'pos': (-453.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.2777, 0.9607)}
        elif scenario_locale == "small_island":
            if road_id == "int_a_small_island":
                return {"pos": (280.397, 210.259, 35.023), 'rot': None, 'rot_quat': turn_X_degrees((-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542), 110)}
            elif road_id == "ai_1":
                return {"pos": (314.573, 105.519, 37.5), 'rot': None, 'rot_quat': turn_X_degrees((-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542), 155)}
            else:
                return {'pos': (254.77, 233.82, 39.5792), 'rot': None, 'rot_quat': (-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542)}
        elif scenario_locale == "jungle_rock_island":
            return {'pos': (-9.99082, 580.726, 156.72), 'rot': None, 'rot_quat': (-0.0067, 0.0051, 0.6231, 0.7821)}
        elif scenario_locale == 'italy':
            if road_id == "default":
                return {'pos': (729.631,763.917,177.997), 'rot': None, 'rot_quat': (-0.0067, 0.0051, 0.6231, 0.7821)}
            elif road_id == "22486": # looks good for input rect but crashes immediately, narrow road
                return {'pos': (-723.4, 631.6, 266.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)} # slightly ahead of start,crashes immediately
                return {'pos': (-694.47,658.12,267.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)} # start, crashes immediately
            elif road_id == "22700": # narrow
                return {'pos': (-1763.44,-1467.49,160.46), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)} # start, crashes immediately
            elif road_id == "22738": # narrow, crashes almost immediately into left low wall
                return {'pos': (1747.30,-75.16,150.66), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 60)}
            elif road_id == "22752":
                return {'pos': (1733.98,-1263.44,170.05), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22762":
                return {'pos': (1182.45,-1797.33,171.59), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22763":
                return {'pos': (-1367.21,759.51,307.28), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22815":
                return {'pos': (-388.07,-558.00,407.62), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22831":
                return {'pos': (313.95,-1945.95,144.11), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22889":
                return {'pos': (-1349.51,760.28,309.26), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22920":
                return {'pos': (1182.45,-1797.33,171.59), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23106":
                return {'pos': (1182.94,-1801.46,171.71), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23113":
                return {'pos': (-1478.87,25.52,302.22), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23132":
                return {'pos': (1513.50,814.87,148.05), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23193":
                return {'pos': (-1542.79,730.60,149.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23199":
                return {'pos': (1224.43,132.39,255.65), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23209":
                return {'pos': (-578.08,1215.80,162.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 100)}
            elif road_id == "23217":
                return {'pos': (-1253.26,471.33,322.71), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23325":
                return {'pos': (1348.08,834.64,191.01), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23398":
                return {'pos': (-1449.63,-302.06,284.02), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23447":
                return {'pos': (-326.59,-838.56,354.59), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23514":
                return {'pos': (-584.13,1209.96,162.43), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23548":
                return {'pos': (927.49,19.38,213.93), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23552":
                return {'pos': (546.33,-871.00,264.50), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23629":
                return {'pos': (100.73,-588.97,331.22), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23665":
                return {'pos': (1265.40,932.52,160.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 150)}
            elif road_id == "23667":
                return {'pos': (-1769.06,-1128.61,142.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23689":
                return {'pos': (276.16,-1435.78,406.12), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23821":
                return {'pos': (-1013.98,402.76,290.71), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23833":
                return {'pos': (-322.82,-1752.61,157.55), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23858":
                return {'pos': (-1896.97,-66.76,148.80), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23898":
                return {'pos': (1718.98,1225.72,217.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -135)}
            elif road_id == "23914":
                return {'pos': (1525.70,813.31,148.26), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23966":
                return {'pos': (1501.21,805.18,147.48), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23978":
                return {'pos': (-1902.02,1567.42,152.42), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23987":
                return {'pos': (-491.94,-770.24,489.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 100)}
            elif road_id == "24007":
                return {'pos': (-68.48,-935.31,463.87), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24029":
                return {'pos': (-585.00,1860.96,152.41), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24043":
                return {'pos': (559.85,884.34,172.88), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 210)}
            # italy roads 400<=length<600
            elif road_id == "21587":
                return {'pos': (265.43,-971.06,270.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "21592":
                return {'pos': (265.78,-890.68,247.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -90)}
            elif road_id == "22518":
                return {'pos': (1754.05,-1284.92,170.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22645":
                return {'pos': (-260.08,395.71,179.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 40)}
            elif road_id == "22654":
                return {'pos': (704.64,212.57,178.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22674":
                return {'pos': (1444.18,-1560.81,164.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22713":
                return {'pos': (1754.70,-1288.44,170.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22744":
                return {'pos': (-456.62,-1355.66,197.83), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "22853":
                return {'pos': (-1753.30,1295.95,139.12), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607),45)}
            elif road_id == "22927":
                return {'pos': (626.50,-1489.26,332.41), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -50)}
            elif road_id == "23037":
                return {'pos': (1105.51,1371.28,139.49), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23103":
                return {'pos': (-1431.80,-253.16,287.63), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23155":
                return {'pos': (-152.73,5.62,259.82), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23189":
                return {'pos': (1754.70,-1288.44,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23197":
                return {'pos': (1754.05,-1284.92,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23210":
                return {'pos': (-1248.98,-1096.07,587.78), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 80)}
            elif road_id == "23289":
                return {'pos': (1444.18,-1560.81,164.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23308":
                return {'pos': (1754.05,-1284.92,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23327":
                return {'pos': (720.06,-886.29,216.04), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23346":
                return {'pos': (1252.73,622.00,201.92), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23497":
                return {'pos': (-948.37,879.28,385.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -150)}
            elif road_id == "23670":
                return {'pos': (102.15,518.06,179.00), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
                # centerline[ 23 ]: [78.65234375, 515.7720947265625, 178.9859619140625]
            elif road_id == "23706":
                return {'pos': (1754.05,-1284.92,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23740":
                return {'pos': (8.14,-557.07,326.49), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 150)}
            elif road_id == "23751":
                return {'pos': (1444.18,-1560.81,164.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23827":
                return {'pos': (-129.24,655.54,192.31), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23949":
                return {'pos': (-1454.84,42.73,303.24), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "23997":
                return {'pos': (431.93,1.94,180.92), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24050":
                return {'pos': (-310.99,-1865.30,160.74), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -98)}
            elif road_id == "24132":
                return {'pos': (-221.87,-1927.10,156.89), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -70)}
            elif road_id == "24217":
                return {'pos': (529.43,616.03,178.00), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24271":
                return {'pos': (1754.70,-1288.44,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24296":
                return {'pos': (-1328.52,1501.30,164.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -65)}
            elif road_id == "24323":
                return {'pos': (64.70,174.23,202.21), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24327":
                return {'pos': (68.60,178.10,202.52), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24332":
                return {'pos': (-1560.53,-255.65,230.70), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24347":
                return {'pos': (1211.34,390.34,235.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24429":
                return {'pos': (1439.79,834.08,208.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24538":
                return {'pos': (884.74,751.11,180.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24597":
                return {'pos': (1324.57,838.79,189.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24619":
                return {'pos': (-336.69,50.83,210.65), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24624":
                return {'pos': (577.46,140.94,183.53), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24642":
                return {'pos': (-1560.53,-255.65,230.70), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24722":
                return {'pos': (-1177.30,-721.74,406.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24878":
                return {'pos': (839.88,1281.43,147.02), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "24931":
                return {'pos': (1440.93,-1559.53,165.00), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 65)}
            elif road_id == "25085":
                return {'pos': (264.33,-586.83,345.14), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25219":
                return {'pos': (464.91,361.13,172.24), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25430":
                return {'pos': (-1114.47,-1533.54,164.19), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25434":
                return {'pos': (571.5, 1234.11, 176.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -215)}
                # 505.523,1254.41,173.275, (-0.02110549248755,0.022285981103778,0.83397573232651,0.55094695091248)
                return {'pos': (569.54,1220.11,176.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -140)}
            elif road_id == "25444":
                return {'pos': (493.77,50.73,187.70), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 50)}
            elif road_id == "25505":
                return {'pos': (344.05,-1454.35,428.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -75)}
            elif road_id == "25509":
                return {'pos': (1259.58,918.19,160.25), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 140)}
            elif road_id == "25511":
                # return {'pos': (567.4, 895.0, 172.83), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 180)}
                return {'pos': (554.4,914.2,170), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 180)}
            elif road_id == "25547":
                return {'pos': (1688.07,-1075.33,152.91), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25555":
                return {'pos': (-1867.60,-130.75,149.08), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25573":
                return {'pos': (-1326.46,1513.28,164.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -50)}
            elif road_id == "25622":
                return {'pos': (-1719.26,106.23,148.42), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25721":
                return {'pos': (-831.32,-1364.53,142.14), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25741":
                return {'pos': (819.14,195.64,186.95), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25893":
                return {'pos': (878.10,745.78,180.86), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "25923":
                return {'pos': (1257.99,1006.8, 175.01), 'rot': None,'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
            elif road_id == "25944":
                return {'pos': (310.03,1802.97,207.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 45)}
            elif road_id == "26124":
                return {'pos': (1182.59,1239.12,148.01), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "26127":
                return {'pos': (-1177.30,-721.74,406.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "26139":
                return {'pos': (1754.05,-1284.92,170.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "26214":
                return {'pos': (-162.74,-155.10,233.43), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "26311":
                return {'pos': (-169.32,-159.67,234.15), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "26360":
                return {'pos': (-1177.30,-721.74,406.85), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "26404":
                return {'pos': (1211.34,390.34,234.94), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -100)}
            elif road_id == "26425":
                return {'pos': (15.93,-1726.24,332.41), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "26464":
                return {'pos': (-479.20,-1624.81,143.42), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -180)}
            elif road_id == "26592":
                return {'pos': (1256.22,915.62,160.28), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "26599":
                return {'pos': (1467.54,-1546.11,163.82), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
            elif road_id == "":
                return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}


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
                road_seg_left.append(self.roadleft[(idx + i) % len(self.roadleft)])
                road_seg_right.append(self.roadright[(idx + i) % len(self.roadright)])
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
                if "platform=" in line:
                    line = line.replace("platform=", "")
                    line = line.split(' ')
                    pos = line[0].split(',')
                    pos = tuple([float(i) for i in pos])
                    rot_quat = line[1].split(',')
                    rot_quat = tuple([float(j) for j in rot_quat])
                    width=float(line[2])
                    cube = ProceduralCube(name='cube_platform',
                                          pos=pos, rot=None, rot_quat=rot_quat,
                                          size=(2, width, 0.5))
                    self.scenario.add_procedural_mesh(cube)
                else:
                    line = line.split(' ')
                    pos = line[0].split(',')
                    pos = tuple([float(i) for i in pos])
                    rot_quat = line[1].split(',')
                    rot_quat = tuple([float(j) for j in rot_quat])
                    self.qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
                    box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
                                         rot_quat=rot_quat, scale=(5,5,5), JBeam='qrbox2', datablock="default_vehicle")
                    self.scenario.add_object(box)


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
        else:
            edges = self.bng.get_road_edges(self.spawnpoint_name)
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
        # print("set up debug line")
        # set up debug line
        for i,p in enumerate(actual_middle[:-1]):
            points.append([p[0], p[1], p[2]])
            point_colors.append([0, 1, 0, 0.1])
            spheres.append([p[0], p[1], p[2], 0.25])
            sphere_colors.append([1, 0, 0, 0.8])
        # print("spawn point:{}".format(spawn))
        # print("beginning of script:{}".format(middle[0]))
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

