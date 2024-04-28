# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from scipy.signal import convolve2d

# Global variables
on_ground = True
height_desired = 1.0
timer = None
startpos = None
timer_done = None

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

def get_potential(map, goal):

    map = map * (-1)

    # Convolve map
    kernel = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], dtype=np.float32)
    kernel /= np.sum(kernel)
    map = convolve2d(map, kernel, mode='same', boundary='fill', fillvalue=0)
    # map = convolve2d(map, kernel, mode='same', boundary='fill', fillvalue=0)

    potential = np.zeros_like(map)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            distance = np.sqrt((i - goal[0]) ** 2 + (j - goal[1]) ** 2)
            potential[i, j] = distance + map[i, j] * 1
    return potential


def get_action(pos, potential):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            x = np.clip(pos[0] + i, 0, potential.shape[0] - 1)
            y = np.clip(pos[1] + j, 0, potential.shape[1] - 1)
            neighbors.append((x, y))

    # Return neighbor with lowest potential
    neighbors_potential = [potential[point[0], point[1]] for point in neighbors]
    min_index = np.argmin(neighbors_potential)
    return neighbors[min_index]

def find_path(pos, goal, potential):
    pos_h = []
    for i in range(10):
        pos = get_action(pos, potential)
        pos_h.append(pos)
    return pos_h

def get_command(sensor_data, camera_data, dt):
    global on_ground, startpos, timer

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)
    
    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]    
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    on_ground = False
    map = occupancy_map(sensor_data)

    # Get the goal position and drone position
    goal = np.array([4.5, 1.5]) / res_pos
    pos = [int(sensor_data['x_global'] / res_pos), int(sensor_data['y_global'] / res_pos)]

    potential = get_potential(map, goal)
    path = find_path(pos, goal, potential)

    if t % 200 == 0:
        plt.imshow(potential, cmap='gray', origin='lower')
        for point in path:
            plt.plot(point[1], point[0], 'ro')
        plt.plot(pos[1], pos[0], 'bo')
        plt.plot(goal[1], goal[0], 'go')
        plt.show()

    next_pos = path[4]
    next_yaw = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
    set_point = [next_pos[0] * res_pos, next_pos[1] * res_pos, next_yaw]

    control_command = go_to_point(set_point, height_desired, sensor_data, dt)
    return control_command # [vx, vy, alt, yaw_rate]




def go_to_point(set_point, set_z, sensor_data, dt):
    drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['yaw']])

    # Set point in the global frame
    set_pos = np.array([set_point[0], set_point[1]])

    # Rotate the set point to the drone frame
    yaw = 0
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    set_pos_body = np.dot(R_yaw, set_pos)
    set_point_body = [set_pos_body[0], set_pos_body[1], set_point[2]]

    pos_error = set_point_body - drone_pos
    pos_error[2] = clip_angle(pos_error[2])

    # P controller
    kp = np.array([1.0, 1.0, 0.5])
    control_command = kp * pos_error
    control_command = np.clip(control_command, -2, 2)
    control_command = [control_command[0], control_command[1], set_z, control_command[2]]

    print("Set point: ", np.array(set_point).round(2))
    print("Drone pos: ", drone_pos.round(2))
    return control_command




# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 5.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.2 # meter
conf = 0.2 # certainty given by each measurement
t = 0 # only for plotting

map = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied

def occupancy_map(sensor_data):
    global map, t
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']
    
    for j in range(4): # 4 sensors
        yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']
        
        for i in range(int(range_max/res_pos)): # range is 2 meters
            dist = i*res_pos
            idx_x = int(np.round((pos_x - min_x + dist*np.cos(yaw_sensor))/res_pos,0))
            idx_y = int(np.round((pos_y - min_y + dist*np.sin(yaw_sensor))/res_pos,0))

            # make sure the current_setpoint is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break
    
    map = np.clip(map, -1, 1) # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == 0:
        plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        plt.savefig("map.png")
        plt.close()
    t +=1
    return map


# Control from the exercises
index_current_setpoint = 0
def path_to_setpoint(path,sensor_data,dt):
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
        control_command = [startpos[0], startpos[1], startpos[2]-0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer,1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']
    distance_drone_to_goal = np.linalg.norm([current_setpoint[0] - x_drone, current_setpoint[1] - y_drone, current_setpoint[2] - z_drone, clip_angle(current_setpoint[3]) - clip_angle(yaw_drone)])

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
    angle = angle%(2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi
    if angle < -np.pi:
        angle += 2*np.pi
    return angle