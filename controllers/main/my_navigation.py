import numpy as np
from scipy.signal import convolve2d


class Navigation:
    startpos = None
    current_pos = None
    fsm_states = ['start', 'going_back']
    fsm_state: str = "start"
    potential_field: np.ndarray = None
    goals: np.array = None
    res_pos = 0.2

    @classmethod
    def update_fsm_state(cls, landing_zone: bool, turned: bool):
        if landing_zone and cls.fsm_state == "start":
            cls.fsm_state = "turning"
        if turned and cls.fsm_state == "turning":
            cls.fsm_state = "going_back"
        if cls.fsm_state == "going_back" and np.linalg.norm(cls.current_pos - cls.startpos) < 0.2:
            cls.fsm_state = "above_start"
        if cls.fsm_state == "above_start" and np.linalg.norm(cls.current_pos - cls.startpos) < 0.05:
            cls.fsm_state = "land"

        print('Objective', cls.current_pos, cls.startpos, np.linalg.norm(cls.current_pos - cls.startpos))
        cls.update_fsm_data()

    @classmethod
    def update_fsm_data(cls):
        if cls.fsm_state == "start":
            cls.goals = [[4.5 / cls.res_pos, i] for i in range(int(3 / cls.res_pos))] + \
                        [[4.5 / cls.res_pos - 1, i] for i in range(int(3 / cls.res_pos))] + \
                        [[4.5 / cls.res_pos + 1, i] for i in range(int(3 / cls.res_pos))]
            cls.goals = np.array(cls.goals)
            cls.goals = cls.goals[abs(cls.goals[:, 1] - 1.5 / cls.res_pos) > 0.75 / cls.res_pos]

        if cls.fsm_state == "going_back" or cls.fsm_state == "above_start":
            cls.goals = [[cls.startpos[0] / cls.res_pos, cls.startpos[1] / cls.res_pos]]

        if cls.fsm_state == "land" or cls.fsm_state == "turning":
            cls.goals = [[cls.current_pos[0] / cls.res_pos, cls.current_pos[1] / cls.res_pos]]


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
                inv_distances = 15 / (1 + distances)
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
                neighbors.append((x, y))

        # Shuffle neighbors
        # neighbors = np.random.permutation(neighbors).tolist()

        # Return neighbor with the highest potential
        neighbors_potential = [cls.potential_field[point[0], point[1]] for point in neighbors]
        min_index = np.argmax(neighbors_potential)
        return neighbors[min_index]

    @classmethod
    def get_path(cls, start_pos: tuple[int, int]) -> np.ndarray:
        path = []
        current_pos = start_pos
        for _ in range(len(cls.potential_field)):
            next_pos = cls.get_action(current_pos)
            path.append(next_pos)
            current_pos = next_pos
        return np.array(path)

    @classmethod
    def navigate(cls, start_pos: tuple[int, int], occupancy_map: np.ndarray) -> np.ndarray:
        """
        Navigate to the goal
        :param start_pos: starting position
        :param goals: np.array(n, 2) goals
        :param occupancy_map: np.ndarray(2D) -1 for obstacles, 1 for free space
        :return: list of positions
        """

        cls.potential_field = cls.get_potential(occupancy_map)
        return cls.get_path(start_pos)
