import time
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


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

        print('Objective', cls.current_pos, cls.startpos, np.linalg.norm(cls.current_pos - cls.startpos))

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

