# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import scipy.interpolate
from scipy.signal import convolve2d
from scipy.interpolate import interp1d

import time
import numpy as np
import matplotlib.pyplot as plt

# Global variables
on_ground = True
height_desired = 1.0
h = 1.0
timer = None
startpos = None
timer_done = None
next_pos = None
nav_range = 0.3


# All available ground truth measurements can be accessed by calling sensor_data[item], where "item" can take the following values:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "v_forward": Forward velocity (body frame)
# "v_left": Leftward velocity (body frame)
# "v_down": Downward velocity (body frame)
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration
# "range_front": Front range finder distance
# "range_down": Donward range finder distance
# "range_left": Leftward range finder distance 
# "range_back": Backward range finder distance
# "range_right": Rightward range finder distance
# "range_down": Downward range finder distance
# "rate_roll": Roll rate (rad/s)
# "rate_pitch": Pitch rate (rad/s)
# "rate_yaw": Yaw rate (rad/s)

# This is the main function where you will implement your control algorithm

def get_command(sensor_data, camera_data, dt):
    global on_ground, startpos, timer, next_pos, h

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]
        Navigation.startpos = np.array(startpos[:2])
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        next_pos = np.array([sensor_data['x_global'], sensor_data['y_global']])
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    on_ground = False
    map = occupancy_map(sensor_data)

    pos_real = [sensor_data['x_global'], sensor_data['y_global']]
    Navigation.current_pos = np.array(pos_real)
    landing_zone = 3.5 < pos_real[0] < 5.0 and 0.0 < pos_real[1] < 3.0
    turned = abs(clip_angle(sensor_data['yaw'] - np.pi)) < 0.3

    # Get the goal position and drone position
    pos = (int(sensor_data['x_global'] / res_pos + 0.5), int(sensor_data['y_global'] / res_pos + 0.5))
    ground_data = sensor_data['z_global'] - sensor_data['range_down']
    Navigation.update_fsm_state(landing_zone, turned, sensor_data['range_down'])

    # Only keep negative values of the map
    negative_map = map
    path = Navigation.navigate(pos, negative_map, ground_data)

    if t % 2000 == 0 and False or (Navigation.fsm_state == "search_padhhdz" and t % 500 ==0): 
        potential = Navigation.potential_field.copy()
        fig = plt.figure('Potential Field Navigation')
        plt.imshow(np.flip(potential, axis=1), cmap='gray', origin='lower')
        plt.plot((3 / res_pos) - path[:, 1] - 1, path[:, 0], 'ro-')
        plt.plot((3 / res_pos) - pos[1] - 1, pos[0], 'bo')
        plt.colorbar()

        if Navigation.fsm_state == "search_pad":
            fig = plt.figure('Ground Map')
            plt.imshow(np.flip(Navigation.ground_map, axis=1), cmap='gray', origin='lower')
            for g in Navigation.goals:
                plt.plot((3 / res_pos) - g[1] - 1, g[0], 'go-')
            plt.plot((3 / res_pos) - path[:, 1] - 1, path[:, 0], 'ro-')
            plt.plot((3 / res_pos) - pos[1] - 1, pos[0], 'bo')
            plt.colorbar()
        plt.show()

    path = path * res_pos

    if Navigation.fsm_state != "hjwkcqh":
        # Interpolate between the points
        path_diff = np.diff(path, axis=0)
        path_diff = np.insert(path_diff, 0, [0, 0], axis=0)
        path_distance = np.linalg.norm(path_diff, axis=1)
        path_cumulative_distance = np.cumsum(path_distance)

        if path_cumulative_distance[-1] < nav_range or len(path) < 3:
            new_pos = path[1] if len(path) > 1 else path[0]
            # print("Could not interpolate", path)
        else:
            next_x = interp1d(path_cumulative_distance, path[:, 0], kind='quadratic')
            next_y = interp1d(path_cumulative_distance, path[:, 1], kind='quadratic')
            new_pos = np.array([next_x(nav_range), next_y(nav_range)])
    else:
        new_pos = path[1]

    next_pos = 0.01 * new_pos + 0.99 * next_pos
    set_point = [next_pos[0], next_pos[1], 0]
    # print("Path", list(path[:3]))
    if Navigation.fsm_state == "start" or Navigation.fsm_state == None:
        next_yaw = np.sin(t / 300) * np.pi / 4
        set_point = [set_point[0], set_point[1], next_yaw]
        h = height_desired
    elif Navigation.fsm_state == "search_pad":
        next_yaw = np.sin(t / 300) * np.pi / 4
        set_point = [set_point[0], set_point[1], next_yaw]
        h = height_desired
        # print("search_pad", sensor_data['z_global'], sensor_data['range_down'])
    elif Navigation.fsm_state == "pad_found":
        set_point = [Navigation.pad_pos[0] * res_pos, Navigation.pad_pos[1] * res_pos, 0]
        h *= 0.995
        # print("pad_found", sensor_data['z_global'], sensor_data['range_down'])
    elif Navigation.fsm_state == "landed":
        set_point = [Navigation.pad_pos[0], Navigation.pad_pos[1], 0]
        h = height_desired      

    elif Navigation.fsm_state == "turning":
        set_point = [pos_real[0], pos_real[1], np.pi]
        h = height_desired
    elif Navigation.fsm_state == "going_back":
        next_yaw = clip_angle((np.sin(t / 300) * np.pi / 4) + np.pi)
        set_point = [set_point[0], set_point[1], next_yaw]
        h = height_desired
    elif Navigation.fsm_state == "above_start":
        set_point = [startpos[0], startpos[1], sensor_data['yaw']]
        h = height_desired
    elif Navigation.fsm_state == "land":
        set_point = [pos_real[0], pos_real[1], sensor_data['yaw']]
        h *= 0.995
    else:
        raise ValueError("Unknown state")
    

    # set_point = [4, 4, np.pi/2]
    control_command = go_to_point(set_point, h, sensor_data, dt)
    if Navigation.fsm_state == "landed":
        control_command = [0, 0, height_desired, 0]
    # control_command = go_to_point([startpos[0], startpos[1], 0], height_desired, sensor_data, dt)
    return control_command  # [vx, vy, alt, yaw_rate]


def go_to_point(set_point, set_z, sensor_data, dt):
    drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global']])
    drone_yaw = sensor_data['yaw']

    set_pos = np.array([set_point[0], set_point[1]])
    set_yaw = set_point[2]

    pos_error = set_pos - drone_pos
    yaw_error = clip_angle(set_yaw - drone_yaw)

    # Rotate the set point to the drone frame
    R_yaw = np.array([[np.cos(drone_yaw), np.sin(drone_yaw)], [-np.sin(drone_yaw), np.cos(drone_yaw)]])
    pos_error = np.dot(R_yaw, pos_error)

    point_error = np.array([pos_error[0], pos_error[1], yaw_error])

    # P controller
    kp = np.array([1.2, 1.2, 0.4])
    # if Navigation.fsm_state == "going_back" or Navigation.fsm_state == "above_start" or Navigation.fsm_state == "land":
    #     kp = np.array([-1.0, -1.0, 0.4])
    control_command = kp * point_error
    control_command = np.clip(control_command, -2, 2)
    control_command = [control_command[0], control_command[1], set_z, control_command[2]]

    # print("Navigation fsm state: ", Navigation.fsm_state)
    # print("Set point: ", np.array(set_point).round(2))
    # print("Drone pos: ", np.array([drone_pos[0], drone_pos[1], drone_yaw]).round(2))
    # print("Error: ", np.array(point_error).round(2), np.linalg.norm(pos_error))
    return control_command


# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0  # meter
min_y, max_y = 0, 3.0  # meter
range_max = 2.0  # meter, maximum range of distance sensor
res_pos = 0.2  # meter
conf = 0.2  # certainty given by each measurement
t = 0  # only for plotting

map = np.zeros((int((max_x - min_x) / res_pos), int((max_y - min_y) / res_pos)))  # 0 = unknown, 1 = free, -1 = occupied


def occupancy_map(sensor_data):
    global map, t
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']

    for j in range(4):  # 4 sensors
        yaw_sensor = yaw + j * np.pi / 2  #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']

        for i in range(int(range_max / res_pos)):  # range is 2 meters
            dist = i * res_pos
            idx_x = int(np.round((pos_x - min_x + dist * np.cos(yaw_sensor)) / res_pos, 0))
            idx_y = int(np.round((pos_y - min_y + dist * np.sin(yaw_sensor)) / res_pos, 0))

            # make sure the current_setpoint is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break

    map = np.clip(map, -1, 1)  # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == -1:
        plt.imshow(np.flip(map, 1), vmin=-1, vmax=1, cmap='gray',
                   origin='lower')  # flip the map to match the coordinate system
        plt.savefig("map.png")
        plt.close()
    t += 1
    return map


# Control from the exercises
index_current_setpoint = 0


def path_to_setpoint(path, sensor_data, dt):
    global on_ground, height_desired, index_current_setpoint, timer, timer_done, startpos

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]
    if on_ground and sensor_data['z_global'] < 0.49:
        current_setpoint = [startpos[0], startpos[1], height_desired, 0.0]
        return current_setpoint
    else:
        on_ground = False

    # Start timer
    if (index_current_setpoint == 1) & (timer is None):
        timer = 0
        print("Time recording started")
    if timer is not None:
        timer += dt
    # Hover at the final setpoint
    if index_current_setpoint == len(path):
        # Uncomment for KF
        control_command = [startpos[0], startpos[1], startpos[2] - 0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer, 1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], \
        sensor_data['yaw']
    distance_drone_to_goal = np.linalg.norm(
        [current_setpoint[0] - x_drone, current_setpoint[1] - y_drone, current_setpoint[2] - z_drone,
         clip_angle(current_setpoint[3]) - clip_angle(yaw_drone)])

    # When the drone reaches the goal setpoint, e.g., distance < 0.1m
    if distance_drone_to_goal < 0.1:
        # Select the next setpoint as the goal position
        index_current_setpoint += 1
        # Hover at the final setpoint
        if index_current_setpoint == len(path):
            current_setpoint = [0.0, 0.0, height_desired, 0.0]
            return current_setpoint

    return current_setpoint


def clip_angle(angle):
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle



class Navigation:
    startpos = None
    current_pos = None
    fsm_states = ['start', 'going_back']
    fsm_state: str = None
    potential_field: np.ndarray = None
    goals: np.array = None
    res_pos = 0.2

    landed_coutdown = 1000

    ground_map: np.ndarray = np.zeros((int(5 / res_pos), int(3 / res_pos))) + 0.5  # 1 for pad, 0 for ground

    @classmethod
    def update_fsm_state(cls, landing_zone: bool, turned: bool, range_down: float):
        if cls.fsm_state is None:
            cls.fsm_state = "start"
            cls.update_fsm_data()

        if landing_zone and cls.fsm_state == "start":
            cls.fsm_state = "search_pad"
            cls.update_fsm_data()
        
        if cls.fsm_state == "search_pad":
            pad = np.where(cls.ground_map > 0.6)
            if len(pad[0]) > 0:
                cls.pad_pos = (pad[0][0], pad[1][0])
                cls.fsm_state = "pad_found"

        if cls.fsm_state == "pad_found" and cls.landed_coutdown < 0:
            cls.fsm_state = "landed"
            cls.update_fsm_data()

        if cls.fsm_state == "landed" and range_down > 0.48:
            cls.fsm_state = "turning"
            cls.update_fsm_data()


        if turned and cls.fsm_state == "turning":
            cls.fsm_state = "going_back"
            cls.update_fsm_data()

        if cls.fsm_state == "going_back" and np.linalg.norm(cls.current_pos - cls.startpos) < 0.2:
            cls.fsm_state = "above_start"
            cls.update_fsm_data()

        if cls.fsm_state == "above_start" and np.linalg.norm(cls.current_pos - cls.startpos) < 0.05:
            cls.fsm_state = "land"
            cls.update_fsm_data()

        # print('Objective', cls.current_pos, cls.startpos, np.linalg.norm(cls.current_pos - cls.startpos))

    @classmethod
    def update_fsm_data(cls):
        if cls.fsm_state == "start":
            cls.goals = [[int(3.7 / cls.res_pos) + i, j] for i in range(int(1.5 / cls.res_pos)) for j in
                         range(int(1 / cls.res_pos), int(2 / cls.res_pos))]
            cls.goals = np.array(cls.goals)

        if cls.fsm_state == "search_pad":
            cls.goals = [[int(3.5 / cls.res_pos) + i, j] for i in range(int(1.5 / cls.res_pos)) for j in
                         range(0, int(3 / cls.res_pos))]
            cls.goals = np.array(cls.goals)

        if cls.fsm_state == "going_back" or cls.fsm_state == "above_start":
            cls.goals = [[cls.startpos[0] / cls.res_pos, cls.startpos[1] / cls.res_pos]]
            cls.goals = np.array(cls.goals)

        if cls.fsm_state == "turning":
            cls.goals = [[cls.current_pos[0] / cls.res_pos, cls.current_pos[1] / cls.res_pos]]
            cls.goals = np.array(cls.goals)

        if cls.fsm_state == "pad_found" or cls.fsm_state == "landed":
            cls.goals = [[cls.pad_pos[0], cls.pad_pos[1]]]
            cls.goals = np.array(cls.goals)

    @classmethod
    def get_potential(cls, occupancy_map: np.ndarray) -> np.ndarray:
        """
        Create a potential field based on the occupancy_map and goals
        :param occupancy_map: np.ndarray(2D) -1 for obstacles, 1 for free space
        :return: np.ndarray(2D) potential field
        """

        # Convolve occupancy_map
        kernel = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], dtype=np.float32)
        kernel /= np.sum(kernel)
        occupancy_map = convolve2d(occupancy_map, kernel, mode='same', boundary='fill', fillvalue=0)

        potential = np.zeros_like(occupancy_map)
        for i in range(occupancy_map.shape[0]):
            for j in range(occupancy_map.shape[1]):
                distances = np.min(np.linalg.norm(cls.goals - np.array([i, j]), axis=1)) / len(occupancy_map)
                inv_distances = 15 / (0.8 + distances)
                potential[i, j] = inv_distances + occupancy_map[i, j]
        return potential

    @classmethod
    def get_action(cls, start_pos: tuple[int, int]) -> tuple[int, int]:
        """
        Get the next action based on the potential field
        :param start_pos: starting position
        :param potential:
        :return:
        """
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                x = np.clip(start_pos[0] + i, 0, cls.potential_field.shape[0] - 1)
                y = np.clip(start_pos[1] + j, 0, cls.potential_field.shape[1] - 1)
                if i == 0 and j == 0:
                    continue
                neighbors.append((x, y))

        # Shuffle neighbors
        # neighbors = np.random.permutation(neighbors).tolist()

        # if cls.fsm_state == "search_pad":
        #    neighbors = neighbors[::-1]

        # Return neighbor with the highest potential
        neighbors_potential = [cls.potential_field[point[0], point[1]] for point in neighbors]
        min_index = np.argmax(neighbors_potential)
        return neighbors[min_index]

    @classmethod
    def get_path(cls, start_pos: tuple[int, int]) -> np.ndarray:
        path = [start_pos]
        current_pos = start_pos
        for _ in range(len(cls.potential_field)):
            next_pos = cls.get_action(current_pos)
            if next_pos in path:
                break
            path.append(next_pos)
            current_pos = next_pos
        return np.array(path)

    @classmethod
    def navigate(cls, start_pos: tuple[int, int], occupancy_map: np.ndarray, ground_data: float = None) -> np.ndarray:
        """
        Navigate to the goal
        :param start_pos: starting position
        :param goals: np.array(n, 2) goals
        :param occupancy_map: np.ndarray(2D) -1 for obstacles, 1 for free space
        :return: list of positions
        """
        if cls.fsm_state == "pad_found":
            cls.landed_coutdown -= 1

        if cls.fsm_state == "search_pad":
            if cls.ground_occupancy_map(ground_data, start_pos):
                cls.goals = [g for g in cls.goals if g[0] != start_pos[0] or g[1] != start_pos[1]]

                obstacles = np.where(occupancy_map == -1)
                for i in range(len(obstacles[0])):
                    cls.goals = [g for g in cls.goals if g[0] != obstacles[0][i] or g[1] != obstacles[1][i]]

            print("Goals", len(cls.goals))
        cls.potential_field = cls.get_potential(occupancy_map)
        return cls.get_path(start_pos)

    @classmethod
    def ground_occupancy_map(cls, ground_data: float, start_pos: tuple[int, int]) -> bool:
        # measurement =  means ground,  means pad
        measurement = ground_data - 0.1
        idx_x = start_pos[0]
        idx_y = start_pos[1]

        # Update the map
        current_ground = cls.ground_map[idx_x, idx_y]
        cls.ground_map[idx_x, idx_y] += measurement
        cls.ground_map = np.clip(cls.ground_map, 0, 1)  # certainty can never be more than 100%
        return cls.ground_map[idx_x, idx_y] == 0 and current_ground != 0

