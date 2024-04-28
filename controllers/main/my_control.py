# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from scipy.signal import convolve2d
from my_navigation import Navigation

# Global variables
on_ground = True
height_desired = 1.0
h = 1.0
timer = None
startpos = None
timer_done = None
next_pos = None


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
        next_pos = np.array([sensor_data['x_global'] / res_pos, sensor_data['y_global'] / res_pos])
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    on_ground = False
    map = occupancy_map(sensor_data)

    pos_real = [sensor_data['x_global'], sensor_data['y_global']]
    Navigation.current_pos = np.array(pos_real)
    landing_zone = 3.5 < pos_real[0] < 5.0 and 0.0 < pos_real[1] < 3.0
    turned = clip_angle(sensor_data['yaw'] - np.pi) < 0.1

    # Get the goal position and drone position
    pos = (int(sensor_data['x_global'] / res_pos), int(sensor_data['y_global'] / res_pos))

    Navigation.update_fsm_state(landing_zone, turned)
    path = Navigation.navigate(pos, map)

    if t % 500 == 0 and False:
        potential = Navigation.potential_field.copy()
        fig = plt.figure('Potential Field Navigation')
        plt.imshow(np.flip(potential, axis=1), cmap='gray', origin='lower')
        plt.plot(14 - path[:, 1], path[:, 0], 'ro-')
        plt.plot(14 - pos[1], pos[0], 'bo')
        plt.colorbar()
        plt.show()

    path = path * res_pos

    # Interpolate between the points
    augmented_path = np.insert(path, 0, pos_real, axis=0)
    augmented_path_diff = np.diff(augmented_path, axis=0)
    augmented_path_diff = np.insert(augmented_path_diff, 0, [0, 0], axis=0)
    path_distance = np.linalg.norm(augmented_path_diff, axis=1)
    path_cumulative_distance = np.cumsum(path_distance)

    if path_cumulative_distance[-1] < 0.5:
        next_pos = path[0]
        print("Could not interpolate")
    else:
        next_x = np.interp(0.5, path_cumulative_distance, augmented_path[:, 0])
        next_y = np.interp(0.5, path_cumulative_distance, augmented_path[:, 1])
        next_pos = np.array([next_x, next_y])

    next_pos = 0.01 * path[1] + 0.99 * next_pos
    set_point = [next_pos[0], next_pos[1], 0]
    if Navigation.fsm_state == "start":
        next_yaw = np.sin(t / 300) * np.pi / 4
        set_point = [set_point[0], set_point[1], 0]
        h = height_desired
    if Navigation.fsm_state == "turning":
        set_point = [pos_real[0], pos_real[1], np.pi]
        h = height_desired
    if Navigation.fsm_state == "going_back":
        set_point = [set_point[0], set_point[1], np.pi]
        h = height_desired
    if Navigation.fsm_state == "above_start":
        set_point = [startpos[0], startpos[1], sensor_data['yaw']]
        h = height_desired
    if Navigation.fsm_state == "land":
        set_point = [pos_real[0], pos_real[1], sensor_data['yaw']]
        h *= 0.995
    else:
        h = height_desired

    # set_point = [4, 4, np.pi/2]
    control_command = go_to_point(set_point, h, sensor_data, dt)
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
    kp = np.array([1.0, 1.0, 0.4])
    # if Navigation.fsm_state == "going_back" or Navigation.fsm_state == "above_start" or Navigation.fsm_state == "land":
    #     kp = np.array([-1.0, -1.0, 0.4])
    control_command = kp * point_error
    control_command = np.clip(control_command, -2, 2)
    control_command = [control_command[0], control_command[1], set_z, control_command[2]]

    print("Navigation fsm state: ", Navigation.fsm_state)
    print("Set point: ", np.array(set_point).round(2))
    print("Drone pos: ", np.array([drone_pos[0], drone_pos[1], drone_yaw]).round(2))
    print("Error: ", np.array([point_error[0], point_error[1], yaw_error]).round(2))
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
