import center_server as cs
import numpy as np
import global_parameters as gp
import math
import user_correlation as ucorr
import accesspoint as ap
import torch
import time
import copy as cp
from collections import defaultdict, deque
import typing

"""
    1) game.hash_observation:
        return games' current observations' string representation
        dtype = string
    2) game.get_observation:
        return [[num], [num]] observation of games' current state
        dtype = np.ndarray
    3) game.goto_next_state(action):
        take input action and move game to next state
        dtype = None
    4) game.end_game:
        return True if game end
        dtype = bool
    5) game.get_finial_reward:
        return the result reward if game end 
        dtype = float
    6) game.restart:
        restart a new round of game
    7) game.get_streng_observations(pi):
        input the policy
        rotate the observation and corresponding policy for data strengthern
        return Tuples of (observation, pi)
    8) game.convert_hash_observation:
        input observation
        return hashed_observation
        string
    9) game.get_valid_action:
        return valid action
        np.ndarray bool
    10) game.get_board_size:
        return board_x, board_y
        int, int
    11) game.get_action_size:
        return num of possible action
        int
"""


class Decentralized_Game:
    def __init__(self, args):
        self.args = args
        self.uav_list = []
        self.user_list = []
        self.accesspoint_list = []
        self.board_length = gp.LENGTH_OF_FIELD
        self.one_side_length = int(math.floor(gp.ACCESS_POINTS_FIELD - 1) / (2 * gp.SQUARE_STEP))
        self.action_size = gp.NUM_OF_UAV
        # accesspoint 0 - uav 0, 1, 2 ... accesspoint 1 - uav 0, 1, 2 ...
        self.center_server = None

        self.aps_observation = []
        self.order_index = False
        self.action_masks = []
        self.pi_rot_bridge = {}
        self.last_reward = 0

        self.state_buffer = []
        self.available_ap = []
        self.scheduler_buffer = deque([], maxlen=self.args.history_length_scheduler)
        self.reward_buffer = []
        self.frame_reward = []
        self.centerlized_reward = None
        self.original_obs = []
        self.history_buffer_length_accesspoint = args.history_length_accesspoint
        if args.history_length_accesspoint <= 1 and args.previous_action_observable:
            raise ValueError("Illegal setting avaliable previous action with less or equal than 1 history length")
        self.history_buffer_length_scheduler = args.history_length_scheduler
        self.history_step_accesspoint = args.multi_step_accesspoint
        self.history_step_scheduler = args.multi_step_scheduler

        self.num_cluster_level = {}  # only use for fast calculation of v1 observation
        # self.restart()
        self.calculate_action_masks()

    # def __deepcopy__(self, memo):
    #     copied = Centralized_Game()
    #     copied.uav_list = self.uav_list
    #     copied.user_list = self.user_list
    #     copied.accesspoint_list = self.accesspoint_list
    #     copied.board_length = self.board_length
    #     copied.action_masks = self.action_masks
    #     copied.action_size = self.action_size
    #     copied.observation = np.copy(self.observation)
    #     if self.order_index:
    #         copied.order_index = True
    #     else:
    #         copied.order_index = False
    #     copied.pi_rot_bridge = self.pi_rot_bridge
    #     copied.center_server = cp.deepcopy(self.center_server)
    #     return copied

    def calculate_action_masks(self):
        self.action_masks = [np.ones(self.action_size, dtype=bool) for _ in range(gp.NUM_OF_UAV)]
        for act in range(self.action_size):
            onehot = self.deci_to_onehot(act)
            for uavs in onehot:
                self.action_masks[uavs][act] = False
            temp = np.reshape(np.arange(0, len(onehot)),
                              (np.sqrt(len(onehot)).astype(np.int), np.sqrt(len(onehot)).astype(np.int)))
            temp = np.reshape(np.rot90(temp, k=1), (-1))
            new_onehot = [0] * len(onehot)
            for key, indi in enumerate(temp):
                new_onehot[key] = onehot[indi]
            # 0123 -> 2031  rotate 90 degree
            # 2 3   ->   3 1
            # 0 1   ->   2 0
            self.pi_rot_bridge[act] = self.onehot_to_deci(new_onehot)
        # use the mask to filte the uav if uav is not able to access
        self.num_cluster_level = {}
        for num in range(gp.USER_CLUSTER_INDICATOR_LENGTH + 1):
            if num > gp.USER_CLUSTER_INDICATOR_LENGTH:
                self.num_cluster_level[num] = float(1 / gp.USER_CLUSTER_INDICATOR_LENGTH)
            elif num >= 2:
                self.num_cluster_level[num] = 1 / float(np.power(2, int(np.log2(num) /
                                                                        np.log2(gp.USER_CLUSTER_INDICATOR_STEP))))
            elif num == 1:
                self.num_cluster_level[num] = int(num)
            else:
                self.num_cluster_level[num] = 0

    def in_board(self, input_pos):
        if input_pos[0] is None or input_pos[1] is None:
            return False
        if 0 <= input_pos[0] < self.board_length and 0 <= input_pos[1] < self.board_length:
            return True
        return False

    def initialized(self):
        self.uav_list = []
        self.user_list = []
        self.accesspoint_list = []
        self.last_reward = 0
        self.order_index = False

        uav_list = []
        for i in range(0, gp.NUM_OF_UAV):
            uav_list.append(ucorr.UAV(i, np.array([np.random.randint(self.board_length),
                                                   np.random.randint(self.board_length)]),
                                      [self.board_length, self.board_length], gp.UAV_FIELD_OF_VIEW))

        ap_list = []
        for i in np.arange(float(self.board_length % gp.DENSE_OF_ACCESSPOINT) / 2, self.board_length,
                           gp.DENSE_OF_ACCESSPOINT):
            for j in np.arange(float(self.board_length % gp.DENSE_OF_ACCESSPOINT) / 2, self.board_length,
                               gp.DENSE_OF_ACCESSPOINT):
                ap_list.append(ap.Access_Point(gp.LINK_THRESHOLD, int(int(i / gp.DENSE_OF_ACCESSPOINT) *
                                                                      math.ceil(self.board_length /
                                                                                gp.DENSE_OF_ACCESSPOINT) +
                                                                      j / gp.DENSE_OF_ACCESSPOINT),
                                               np.array([i, j])))

        gp.REFRESH_SCALE(self.args.user_cluster_scale)

        ue_list = []
        index = 0
        if gp.NUM_OF_CLUSTER >= 1:  # PCP
            for i in range(0, gp.NUM_OF_UAV):
                uav_pos = np.array(uav_list[i].position)
                for clu in range(gp.NUM_OF_CLUSTER):
                    c_position = np.array([None, None])
                    while not self.in_board(c_position):
                        bias_position = np.array([np.random.randint(gp.CLUSTER_SCALE * 2) - gp.CLUSTER_SCALE,
                                                  np.random.randint(gp.CLUSTER_SCALE * 2) - gp.CLUSTER_SCALE])
                        c_position = uav_pos - bias_position
                    for ue in range(0, np.random.poisson(gp.DENSE_OF_USERS_PCP)):
                        position = np.array([None, None])
                        while not self.in_board(position):
                            bias_position = np.array([np.random.randint(gp.UE_SCALE * 2) - gp.UE_SCALE,
                                                      np.random.randint(gp.UE_SCALE * 2) - gp.UE_SCALE])
                            position = c_position - bias_position
                        ue_list.append(ucorr.User_VR(position, [self.board_length, self.board_length], index,
                                                     np.array([int(np.random.rand() * gp.UAV_FIELD_OF_VIEW[0]),
                                                               int(np.random.rand() * gp.UAV_FIELD_OF_VIEW[1])]),
                                                     uav_list[i], ap_list, gp.USER_FIELD_OF_VIEW))
                        index += 1
        elif gp.NUM_OF_CLUSTER <= 0:  # random users
            for i in range(0, gp.NUM_OF_UAV):
                for ue in range(0, np.random.poisson(int(gp.DENSE_OF_USERS / gp.NUM_OF_UAV))):
                    ue_list.append(ucorr.User_VR(np.array([np.random.randint(self.board_length),
                                                           np.random.randint(self.board_length)]),
                                                 [self.board_length, self.board_length], index,
                                                 np.array([int(np.random.rand() * gp.UAV_FIELD_OF_VIEW[0]),
                                                           int(np.random.rand() * gp.UAV_FIELD_OF_VIEW[1])]),
                                                 uav_list[i], ap_list, gp.USER_FIELD_OF_VIEW))
                    index += 1

        self.uav_list = uav_list
        self.accesspoint_list = ap_list
        self.user_list = ue_list

        self.center_server = cs.Center_Server(self.user_list, self.accesspoint_list, self.uav_list,
                                              gp.CORRELATION_THRESHOLD, gp.CLUSTERING_METHOD, gp.LINK_THRESHOLD)

        if self.action_size != len(self.uav_list):
            self.action_size = len(self.uav_list)
            self.calculate_action_masks()
            # if approximate num from global parameters directly is wrong, recalculate action size

        self.center_server.initial()
        # ---------reset replay buffer---------#
        self.state_buffer = []
        for _ in range(len(self.accesspoint_list)):
            self.state_buffer.append(deque([], maxlen=self.args.history_length_accesspoint))
        self.available_ap = []
        for _ in range(len(self.accesspoint_list)):
            self.available_ap.append(np.ones(len(self.uav_list), dtype=bool))

        self.reward_buffer = []
        self.frame_reward = []
        for _ in range(len(self.accesspoint_list)):
            self.reward_buffer.append(deque([], maxlen=3))
            for __ in range(3):
                self.reward_buffer[_].append(np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                                       2 * np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]))
        self.centerlized_reward = np.zeros(len(self.user_list))
        self.center_server.centerlized_reward = np.zeros(len(self.user_list))
        self.scheduler_buffer.clear()
        boardx, boardy, dims = self.get_board_size()

        self.original_obs = []

        for _ in range(self.history_buffer_length_scheduler):
            if gp.GOP >= 2:
                self.scheduler_buffer.append(
                    torch.zeros(math.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT) * gp.UAV_FIELD_OF_VIEW[0] * 2,
                                math.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT) * gp.NUM_OF_UAV *
                                gp.UAV_FIELD_OF_VIEW[1],
                                device=self.args.device))
            else:
                self.scheduler_buffer.append(
                    torch.zeros(math.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT) * gp.UAV_FIELD_OF_VIEW[0],
                                math.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT) * gp.NUM_OF_UAV *
                                gp.UAV_FIELD_OF_VIEW[1],
                                device=self.args.device))

        for index in range(len(self.accesspoint_list)):
            for _ in range(self.history_buffer_length_accesspoint):
                self.state_buffer[index].append(torch.zeros(dims * int(self.one_side_length * 2 + 1),
                                                            int(self.one_side_length * 2 + 1), device=self.args.device))

    def reset(self):
        np.random.seed(int(time.time() % 1 * 10e8))
        self.initialized()
        self.scheduler_buffer.append(self.obtain_popularity_count())
        return torch.stack(list(self.scheduler_buffer), dim=0)

    def get_board_size(self):
        return int(self.one_side_length * 2 + 1), int(self.one_side_length * 2 + 1), \
               len(self.uav_list) * gp.OBSERVATION_DIMS

    def get_resource_action_space(self):
        if gp.GOP >= 2:
            return gp.TOTAL_NUM_TILES * len(self.uav_list) * 2, gp.UAV_FIELD_OF_VIEW
        else:
            return gp.TOTAL_NUM_TILES * len(self.uav_list), gp.UAV_FIELD_OF_VIEW

    def end_game(self):
        if self.center_server.clock >= len(self.center_server.resource_block_allocated):
            return True
        return False

    def get_finial_reward(self):
        if self.end_game():
            # reward = self.center_server.count_performance()  # reward is np.ndarray(len: users number)
            # reward = reward / 10 - 0.5
            # mean_reward = np.mean(reward)
            # variance_reward = np.var(reward)
            # return mean_reward - variance_reward
            # # reward = np.mean(reward/10 - 1)
            # # version_1 reward
            return np.mean(np.sum(self.center_server.psnr_count, axis=1))
        else:
            raise ValueError("Game should end before call this function")

    def get_accumu_reward_linear(self):
        reward = self.center_server.count_instant_performance_linear()  # reward is np.ndarray(len: users number)
        mean_reward = np.mean(reward)
        variance_reward = np.var(reward)
        current_reward = mean_reward - variance_reward
        delta_reward = current_reward - self.last_reward
        self.last_reward = current_reward
        return delta_reward
        # reward = np.mean(reward/10 - 1)
        # version_1 reward

    def get_action_size(self):
        return self.action_size

    def plot_grid_map(self, user_list):
        grid_map = np.zeros([int(self.board_length / gp.SQUARE_STEP), int(self.board_length / gp.SQUARE_STEP)],
                            dtype=bool)
        clusters_locations = np.array([self.user_list[ues].position for ues in user_list])
        clusters_locations_norms = np.floor(clusters_locations / gp.SQUARE_STEP).astype(int)
        for locations in clusters_locations_norms:
            grid_map[locations[0], locations[1]] = True
        return grid_map

    def plot_grid_map_weighted(self, uav_id, cluster_list):
        current_sphere_id = uav_id
        grid_map = np.zeros([int(self.board_length / gp.SQUARE_STEP), int(self.board_length / gp.SQUARE_STEP)])
        clusters_locations = np.array([ues.position for ues in self.user_list])
        clusters_locations_norms = np.floor(clusters_locations / gp.SQUARE_STEP).astype(int)
        for cluster in cluster_list:
            if self.user_list[cluster[0]].sphere_id == current_sphere_id:
                grid_map_cluster = np.zeros([int(self.board_length / gp.SQUARE_STEP),
                                             int(self.board_length / gp.SQUARE_STEP)],
                                            dtype=bool)
                for ues in cluster:
                    grid_map_cluster[clusters_locations_norms[ues][0],
                                     clusters_locations_norms[ues][1]] = True
                grid_map += grid_map_cluster.astype(np.int)
        grid_map = grid_map / gp.USER_CLUSTER_INDICATOR_LENGTH
        grid_map[grid_map > 1] = 1
        # inverse version
        # grid_map = (1 + gp.USER_CLUSTER_INDICATOR_LENGTH - grid_map) / gp.USER_CLUSTER_INDICATOR_LENGTH
        # grid_map[grid_map <= 0] = 1 / gp.USER_CLUSTER_INDICATOR_LENGTH
        # grid_map[grid_map > 1] = 0
        return grid_map

    def plot_grid_map_linear_psnr_weighted(self, uav_id, psnr_result):
        """
        :parameter
        psnr_result:
                psnr of current users, must with length of len(self.user_list)
        uav_id:
                The target uav
        """
        grid_map = np.zeros([int(self.board_length / gp.SQUARE_STEP), int(self.board_length / gp.SQUARE_STEP)])
        grid_num_map = np.zeros([int(self.board_length / gp.SQUARE_STEP), int(self.board_length / gp.SQUARE_STEP)])
        clusters_locations = np.array([ues.position for ues in self.user_list])
        clusters_locations_norms = np.floor(clusters_locations / gp.SQUARE_STEP).astype(int)
        for ue, psnr, location in zip(self.user_list, psnr_result, clusters_locations_norms):
            if ue.sphere_id == uav_id:
                grid_map[location[0], location[1]] += 1 - psnr
                grid_num_map[location[0], location[1]] += 1
        # TODO: Change this "1" if want to use another type of psnr value
        grid_num_map[grid_num_map == 0] += 1
        grid_map = grid_map / grid_num_map
        # inverse version
        # grid_map = (1 + gp.USER_CLUSTER_INDICATOR_LENGTH - grid_map) / gp.USER_CLUSTER_INDICATOR_LENGTH
        # grid_map[grid_map <= 0] = 1 / gp.USER_CLUSTER_INDICATOR_LENGTH
        # grid_map[grid_map > 1] = 0
        return grid_map

    def get_observation_tensor(self):
        """:return List of tensor"""
        if not self.order_index:
            self.get_observation()
        return [torch.tensor(aps_obv, dtype=torch.float32, device=self.args.device) for aps_obv in self.aps_observation]

    @staticmethod
    def pad_with_zeros(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    # def rotate_obs(self, res_obs, aps, dims=(1,2)):
    #     id = aps.id
    #     x = (id // gp.ACCESS_POINT_PER_EDGE) < (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     y = (id % gp.ACCESS_POINT_PER_EDGE) < (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     z = (id // gp.ACCESS_POINT_PER_EDGE) <= (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     w = (id % gp.ACCESS_POINT_PER_EDGE) <= (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     m = (id % gp.ACCESS_POINT_PER_EDGE) == (gp.ACCESS_POINT_PER_EDGE - 1) / 2 and \
    #         (id // gp.ACCESS_POINT_PER_EDGE) == (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     if not w and z or m:
    #         return res_obs
    #     elif x and w:  # dimension 2
    #         return np.rot90(res_obs, 3, dims)
    #     elif not x and y:  # dimension 3
    #         return np.rot90(res_obs, 2, dims)
    #     elif not y and not z:  # dimesion 4
    #         return np.rot90(res_obs, 1, dims)
    #     else:
    #         raise ValueError("Dimension Error")
    #
    # def rotate_single(self, res, aps):
    #     if res.shape.__len__() > 2:
    #         raise IndexError("Not suitable shape")
    #     id = aps.id
    #     x = (id // gp.ACCESS_POINT_PER_EDGE) < (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     y = (id % gp.ACCESS_POINT_PER_EDGE) < (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     z = (id // gp.ACCESS_POINT_PER_EDGE) <= (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     w = (id % gp.ACCESS_POINT_PER_EDGE) <= (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     m = (id % gp.ACCESS_POINT_PER_EDGE) == (gp.ACCESS_POINT_PER_EDGE - 1) / 2 and \
    #         (id // gp.ACCESS_POINT_PER_EDGE) == (gp.ACCESS_POINT_PER_EDGE - 1) / 2
    #     if not w and z or m:
    #         return res
    #     elif x and w:  # dimension 2
    #         return torch.rot90(res, 3, (0, 1))
    #     elif not x and y:  # dimension 3
    #         return torch.rot90(res, 2, (0, 1))
    #     elif not y and not z:  # dimesion 4
    #         return torch.rot90(res, 1, (0, 1))
    #     else:
    #         raise ValueError("Dimension Error")

    def get_observation(self):
        if self.order_index:
            return self.aps_observation

        if gp.OBSERVATION_VERSION == 1:
            obs = self._get_observation_v1()
        elif gp.OBSERVATION_VERSION == 2:
            obs = self._get_observation_v2()
        elif gp.OBSERVATION_VERSION == 3:
            obs = self._get_observation_v3()
        elif gp.OBSERVATION_VERSION == 4:
            obs = self._get_observation_v4()
        elif gp.OBSERVATION_VERSION == 5:
            obs = self._get_observation_v5()
        elif gp.OBSERVATION_VERSION == 6:
            obs = self._get_observation_v6()
        elif gp.OBSERVATION_VERSION == 7 and gp.OBSERVATION_DIMS == 5:
            obs = self._get_observation_v7()
        elif gp.OBSERVATION_VERSION == 7 and gp.OBSERVATION_DIMS == 4:
            obs = self._get_observation_v7_1()
        elif gp.OBSERVATION_VERSION == 7 and gp.OBSERVATION_DIMS == 3:
            obs = self._get_observation_v7_2()
        elif gp.OBSERVATION_VERSION == 8:
            obs = self._get_observation_v8()
        else:
            raise ValueError("Illegal observation version")

        if (gp.ACCESS_POINTS_FIELD - 1) % (2 * gp.SQUARE_STEP) != 0:
            raise ValueError("Access point field must be odd and diviable by step size")

        pad_width = math.floor(1 + ((gp.ACCESS_POINTS_FIELD - 1) / 2 -
                                    (gp.LENGTH_OF_FIELD // gp.DENSE_OF_ACCESSPOINT) / 2) / gp.SQUARE_STEP)

        obs_decentral = []
        for index in range(len(self.uav_list)):
            temp = []
            for index_obs in range(gp.OBSERVATION_DIMS):
                temp.append(np.pad(obs[index][index_obs], int(pad_width), self.pad_with_zeros, padder=0))
            obs_decentral.append(np.stack(temp, axis=0))

        for index in range(len(self.accesspoint_list)):
            self.available_ap[index] = np.zeros(len(self.uav_list), dtype=bool)

        aps_observation = []
        for ap_index, aps in enumerate(self.accesspoint_list):
            temp = []
            reward = []
            for index in range(len(self.uav_list)):
                a = math.floor(aps.position[0] / gp.SQUARE_STEP) + pad_width - self.one_side_length
                b = math.floor(aps.position[0] / gp.SQUARE_STEP) + pad_width + self.one_side_length + 1
                c = math.floor(aps.position[1] / gp.SQUARE_STEP) + pad_width - self.one_side_length
                d = math.floor(aps.position[1] / gp.SQUARE_STEP) + pad_width + self.one_side_length + 1
                res_obs = obs_decentral[index][:, int(a):int(b), int(c):int(d)]

                # res_obs = self.rotate_obs(res_obs, aps, dims=(1,2))  # rotate everything to dimension 1

                reward.append(res_obs[-1])
                if res_obs[-1].any():
                    self.available_ap[ap_index][index] = True
                    temp.append(res_obs)
                elif gp.NULL_UNAVALIABLE_UAV:
                    temp.append(res_obs * 0)  # Null all elements for non avaliable uav
                else:
                    temp.append(res_obs)
            self.reward_buffer[ap_index].append(np.concatenate(reward, axis=1))
            if gp.OBSERVATION_VERSION <= 7:
                aps_observation.append(np.concatenate(temp, axis=0).reshape(-1, temp[0].shape[2]))
            elif gp.OBSERVATION_VERSION == 8:
                aps_observation.append(np.concatenate(temp, axis=0))
        self.aps_observation = aps_observation
        # list ap: list uav: ndarray observation
        return self.aps_observation

    def _get_observation_v1(self):
        """
        :return Observation which is a 5*5*3 matrix with position of uav, postion of user (0/1), number of clusters
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [(np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.ones([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                 np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=int))
                       for _ in range(len(self.uav_list))]
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                num = self.center_server.num_of_clusters_in_each_uav[uav.id]
                users = self.plot_grid_map(self.center_server.uav_user_match_table[uav.id]).astype(np.int)
            else:
                num = 0
                users = observation[uav.id][1]

            if num > gp.USER_CLUSTER_INDICATOR_LENGTH:
                num_scale = float(1 / gp.USER_CLUSTER_INDICATOR_LENGTH)
            else:
                num_scale = self.num_cluster_level[num]

            observation[uav.id] = np.stack((observation[uav.id][0], users, users.astype(np.int) * num_scale), axis=0)
        self.observation = observation
        return self.observation

    def _get_observation_v2(self):
        """
        :return Observation which is a 5*5*3 matrix with position of uav, ue position in largest cluster, second cluster
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [(np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=int))
                       for _ in range(len(self.uav_list))]
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                cluster_uav_ue = [clus for clus in self.center_server.cluster_result
                                  if self.user_list[clus[0]].sphere_id == uav.id]
                cluster_uav_ue = sorted(cluster_uav_ue, key=lambda cls: len(cls), reverse=True)
                if len(cluster_uav_ue) >= 2:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
                    sec_largest_clust = self.plot_grid_map(cluster_uav_ue[1]).astype(np.int)
                else:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
                    sec_largest_clust = observation[uav.id][2]
            else:
                largest_clust = observation[uav.id][1]
                sec_largest_clust = observation[uav.id][2]

            observation[uav.id] = np.stack((observation[uav.id][0], largest_clust, sec_largest_clust), axis=0)
        self.observation = observation
        return self.observation

    # observation: uav_id: (uav_location, users grid, num grid)

    def _get_observation_v3(self):
        """
        :return Observation which is a 5*5*3 matrix with position of uav, ue position in largest cluster, total position
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [(np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=int))
                       for _ in range(len(self.uav_list))]
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                users = self.plot_grid_map(self.center_server.uav_user_match_table[uav.id]).astype(np.int)
            else:
                users = observation[uav.id][2]

            if uav.id in self.center_server.uav_user_match_table:
                cluster_uav_ue = [clus for clus in self.center_server.cluster_result
                                  if self.user_list[clus[0]].sphere_id == uav.id]
                cluster_uav_ue = sorted(cluster_uav_ue, key=lambda cls: len(cls), reverse=True)
                if len(cluster_uav_ue) >= 2:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
                else:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
            else:
                largest_clust = observation[uav.id][1]

            observation[uav.id] = np.stack((observation[uav.id][0], largest_clust, users), axis=0)
        self.observation = observation
        return self.observation
        # observation: uav pos, largest cluster, total cluster

    def _get_observation_v4(self):
        """
        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [(np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=int))
                       for _ in range(len(self.uav_list))]
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            for aps in self.accesspoint_list:
                aps_pos = np.floor(aps.position / gp.SQUARE_STEP).astype(int)
                observation[uav.id][1][aps_pos[0], aps_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                users = self.plot_grid_map_weighted(uav.id, self.center_server.cluster_result)
            else:
                users = observation[uav.id][3]

            if uav.id in self.center_server.uav_user_match_table:
                cluster_uav_ue = [clus for clus in self.center_server.cluster_result
                                  if self.user_list[clus[0]].sphere_id == uav.id]
                cluster_uav_ue = sorted(cluster_uav_ue, key=lambda cls: len(cls), reverse=True)
                if len(cluster_uav_ue) >= 2:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
                else:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
            else:
                largest_clust = observation[uav.id][2]

            observation[uav.id] = np.stack((observation[uav.id][0], observation[uav.id][1],
                                            largest_clust, users), axis=0)
        self.observation = observation
        return self.observation

    def _get_observation_v5(self):
        """
        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [(np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]))
                       for _ in range(len(self.uav_list))]
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            for aps in self.accesspoint_list:
                aps_pos = np.floor(aps.position / gp.SQUARE_STEP).astype(int)
                observation[uav.id][1][aps_pos[0], aps_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                users = self.plot_grid_map_weighted(uav.id, self.center_server.cluster_result)
            else:
                users = observation[uav.id][4]

            if uav.id in self.center_server.uav_user_match_table:
                cluster_uav_ue = [clus for clus in self.center_server.cluster_result
                                  if self.user_list[clus[0]].sphere_id == uav.id]
                cluster_uav_ue = sorted(cluster_uav_ue, key=lambda cls: len(cls), reverse=True)
                if len(cluster_uav_ue) >= 2:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
                    second_clust = self.plot_grid_map(cluster_uav_ue[1]).astype(np.int)
                else:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
                    second_clust = observation[uav.id][3]
            else:
                largest_clust = observation[uav.id][2]
                second_clust = observation[uav.id][3]

            observation[uav.id] = np.stack((observation[uav.id][0], observation[uav.id][1],
                                            largest_clust, second_clust, users), axis=0)
        self.observation = observation
        return self.observation

    def _get_observation_v6(self):
        """
        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [(np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]))
                       for _ in range(len(self.uav_list))]
        psnr_list = self.center_server.count_instant_performance_linear()
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            for aps in self.accesspoint_list:
                aps_pos = np.floor(aps.position / gp.SQUARE_STEP).astype(int)
                observation[uav.id][1][aps_pos[0], aps_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                users = self.plot_grid_map_linear_psnr_weighted(uav.id, psnr_list)
            else:
                users = observation[uav.id][4]

            if uav.id in self.center_server.uav_user_match_table:
                cluster_uav_ue = [clus for clus in self.center_server.cluster_result
                                  if self.user_list[clus[0]].sphere_id == uav.id]
                cluster_uav_ue = sorted(cluster_uav_ue, key=lambda cls: len(cls), reverse=True)
                if len(cluster_uav_ue) >= 2:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
                    second_clust = self.plot_grid_map(cluster_uav_ue[1]).astype(np.int)
                else:
                    largest_clust = self.plot_grid_map(cluster_uav_ue[0]).astype(np.int)
                    second_clust = observation[uav.id][3]
            else:
                largest_clust = observation[uav.id][2]
                second_clust = observation[uav.id][3]

            observation[uav.id] = np.stack((observation[uav.id][0], observation[uav.id][1],
                                            largest_clust, second_clust, users), axis=0)
        self.observation = observation
        return self.observation

    def _get_observation_v7(self):
        """
        ATTENTION: ASSIGN NEW POPULARITY BEFORE RUNNING THIS FUNTION!!!!!!!!!!

        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        if len(self.center_server.stack_of_popular) == 0:
            raise IndexError("Nothing in popularity stack")
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [[np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)])]
                       for _ in range(len(self.uav_list))]
        psnr_list = self.center_server.count_instant_performance_linear()
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            for aps in self.accesspoint_list:
                aps_pos = np.floor(aps.position / gp.SQUARE_STEP).astype(int)
                observation[uav.id][1][aps_pos[0], aps_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                users = self.plot_grid_map_linear_psnr_weighted(uav.id, psnr_list)
            else:
                users = observation[uav.id][4]

            if self.center_server.stack_of_popular[uav.id].__len__() > 0:
                scheduled_tile = self.center_server.stack_of_popular[uav.id]
                if scheduled_tile.__len__() > gp.DEFAULT_NUM_OF_RB_PER_RES:
                    scheduled_tile = scheduled_tile[:gp.DEFAULT_NUM_OF_RB_PER_RES]
                for index, tiles in enumerate(scheduled_tile):
                    if index < (len(scheduled_tile) / 2):
                        for user in self.user_list:
                            if user.sphere_id == uav.id and tiles in user.resource[user.transmission_mask]:
                                user_pos = np.floor(user.position / gp.SQUARE_STEP).astype(int)
                                observation[uav.id][2][user_pos[0], user_pos[1]] += 1
                    else:
                        for user in self.user_list:
                            if user.sphere_id == uav.id and tiles in user.resource[user.transmission_mask]:
                                user_pos = np.floor(user.position / gp.SQUARE_STEP).astype(int)
                                observation[uav.id][3][user_pos[0], user_pos[1]] += 1
                users[np.logical_or(observation[uav.id][2], observation[uav.id][3]) == 0] = 0
                observation[uav.id][2] = observation[uav.id][2] / math.ceil(len(scheduled_tile) / 2)
                observation[uav.id][3] = observation[uav.id][3] / math.ceil(len(scheduled_tile) / 2)
            # normalized to 0~1

            observation[uav.id] = np.stack((observation[uav.id][0], observation[uav.id][1],
                                            observation[uav.id][2], observation[uav.id][3], users), axis=0)
        self.observation = observation
        return self.observation

    def _get_observation_v7_1(self):
        """
        ATTENTION: ASSIGN NEW POPULARITY BEFORE RUNNING THIS FUNTION!!!!!!!!!!

        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        if len(self.center_server.stack_of_popular) == 0:
            raise IndexError("Nothing in popularity stack")
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [[np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)])]
                       for _ in range(len(self.uav_list))]
        psnr_list = self.center_server.count_accumu_performance_linear()
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            for aps in self.accesspoint_list:
                aps_pos = np.floor(aps.position / gp.SQUARE_STEP).astype(int)
                observation[uav.id][1][aps_pos[0], aps_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                users = self.plot_grid_map_linear_psnr_weighted(uav.id, psnr_list)
            else:
                users = observation[uav.id][3]

            if self.center_server.stack_of_popular[uav.id].__len__() > 0:
                scheduled_tile = self.center_server.stack_of_popular[uav.id]
                if scheduled_tile.__len__() > gp.DEFAULT_NUM_OF_RB_PER_RES:
                    scheduled_tile = scheduled_tile[:gp.DEFAULT_NUM_OF_RB_PER_RES]
                for index, tiles in enumerate(scheduled_tile):
                    for user in self.user_list:
                        if user.sphere_id == uav.id and tiles in user.resource[user.transmission_mask]:
                            user_pos = np.floor(user.position / gp.SQUARE_STEP).astype(int)
                            observation[uav.id][2][user_pos[0], user_pos[1]] += 1
                # users[observation[uav.id][2] == 0] = 0
                observation[uav.id][2] = observation[uav.id][2] / len(scheduled_tile)
            # normalized to 0~1

            observation[uav.id] = np.stack((observation[uav.id][0], observation[uav.id][1],
                                            observation[uav.id][2], users), axis=0)
        self.observation = observation
        return self.observation

    def _get_observation_v7_2(self):
        """
        ATTENTION: ASSIGN NEW POPULARITY BEFORE RUNNING THIS FUNTION!!!!!!!!!!

        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        if len(self.center_server.stack_of_popular) == 0:
            raise IndexError("Nothing in popularity stack")
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [[np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)])]
                       for _ in range(len(self.uav_list))]
        psnr_list = self.center_server.count_accumu_performance_linear()
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            for aps in self.accesspoint_list:
                aps_pos = np.floor(aps.position / gp.SQUARE_STEP).astype(int)
                observation[uav.id][1][aps_pos[0], aps_pos[1]] = True

            if self.center_server.stack_of_popular[uav.id].__len__() > 0:
                scheduled_tile = self.center_server.stack_of_popular[uav.id]
                if scheduled_tile.__len__() > gp.DEFAULT_NUM_OF_RB_PER_RES:
                    scheduled_tile = scheduled_tile[:gp.DEFAULT_NUM_OF_RB_PER_RES]
                for index, tiles in enumerate(scheduled_tile):
                    for user in self.user_list:
                        if user.sphere_id == uav.id and tiles in user.resource[user.transmission_mask]:
                            user_pos = np.floor(user.position / gp.SQUARE_STEP).astype(int)
                            observation[uav.id][2][user_pos[0], user_pos[1]] += 1
                # users[observation[uav.id][2] == 0] = 0
                observation[uav.id][2] = observation[uav.id][2] / len(scheduled_tile)
            # normalized to 0~1

            observation[uav.id] = np.stack((observation[uav.id][0], observation[uav.id][1],
                                            observation[uav.id][2]), axis=0)
        self.observation = observation
        return self.observation

    def _get_observation_v8(self):
        """
        ATTENTION: ASSIGN NEW POPULARITY BEFORE RUNNING THIS FUNTION!!!!!!!!!!

        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """
        if self.order_index:
            return self.aps_observation
        self.order_index = True
        if len(self.center_server.stack_of_popular) == 0:
            raise IndexError("Nothing in popularity stack")
        # make sure functions are called in correct order

        self.center_server.centralized_clustering()
        observation = [[np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]),
                        np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)])]
                       for _ in range(len(self.uav_list))]
        psnr_list = self.center_server.count_accumu_performance_linear()
        for uav in self.uav_list:
            uav_pos = np.floor(uav.position / gp.SQUARE_STEP).astype(int)
            observation[uav.id][0][uav_pos[0], uav_pos[1]] = True
            for aps in self.accesspoint_list:
                aps_pos = np.floor(aps.position / gp.SQUARE_STEP).astype(int)
                observation[uav.id][1][aps_pos[0], aps_pos[1]] = True
            if uav.id in self.center_server.uav_user_match_table:
                users = self.plot_grid_map_linear_psnr_weighted(uav.id, psnr_list)
            else:
                users = observation[uav.id][3]

            if self.center_server.stack_of_popular[uav.id].__len__() > 0:
                scheduled_tile = self.center_server.stack_of_popular[uav.id]
                if scheduled_tile.__len__() > gp.DEFAULT_NUM_OF_RB_PER_RES:
                    scheduled_tile = scheduled_tile[:gp.DEFAULT_NUM_OF_RB_PER_RES]
                for index, tiles in enumerate(scheduled_tile):
                    for user in self.user_list:
                        if user.sphere_id == uav.id and tiles in user.resource[user.transmission_mask]:
                            user_pos = np.floor(user.position / gp.SQUARE_STEP).astype(int)
                            observation[uav.id][2][user_pos[0], user_pos[1]] += 1
                # users[observation[uav.id][2] == 0] = 0
                observation[uav.id][2] = observation[uav.id][2] / len(scheduled_tile)
            # normalized to 0~1

            observation[uav.id] = np.stack((observation[uav.id][0], observation[uav.id][1],
                                            observation[uav.id][2], users), axis=0)
        self.observation = observation
        return self.observation

    def deci_to_onehot(self, index):
        """
        :parameter index is the ap-uav map [ap-uav-0-0, ap-uav-0-1, ..., ap-uav-3-1] in binary form
                            max_length is uav^ap, convert to uav-based representation
                            2^2 -> binary -> 01 -> ap0-uav0-ap1-uav1
        """
        conv = [0] * len(self.accesspoint_list)
        if index == 0:
            return conv
        for ind in range(len(conv)):
            quotient = index // len(self.uav_list)
            remainder = index % len(self.uav_list)
            conv[ind] = remainder
            if quotient == 0:
                break
            index = quotient
        conv.reverse()
        return conv

    def onehot_to_deci(self, onehot: list):
        onehot.reverse()
        sum = 0
        for step, nums in enumerate(onehot):
            sum += nums * (len(self.uav_list) ** step)
        return sum

    def obtain_decision_from_action(self, action):
        # conv = self.onehot_to_deci(action)
        for index, act in enumerate(action):
            if self.available_ap[index][act]:
                continue
            else:
                if not self.available_ap[index].any():
                    action[index] = -1
                    continue
                # new_num = np.random.randint(0, len(self.uav_list))
                # while not self.available_ap[index][new_num]:
                #     new_num = np.random.randint(0, len(self.uav_list))
                # action[index] = new_num
        self.center_server.uav_ap_match_table.clear()
        self.center_server.ap_user_match_table.clear()
        for ap_index, uav_indices in enumerate(action):
            if uav_indices == -1:
                continue
            self.center_server.uav_ap_match_table[uav_indices] = \
                np.union1d(self.center_server.uav_ap_match_table[uav_indices], np.array([ap_index]))
        for uav, users in self.center_server.uav_user_match_table.items():
            for aps in self.center_server.uav_ap_match_table[uav]:
                self.center_server.ap_user_match_table[aps] = users
        # equal to obtain decision from ap step
        return action

    def obtain_popularity_count(self):
        return torch.tensor(self.center_server.obtain_popularity_count(), dtype=torch.float32, device=self.args.device)

    def remove_previous_action(self, state):
        # TODO: Change this part if the position of ap's postion layer changed in observation
        if 5 <= gp.OBSERVATION_VERSION < 7:
            for uav_index in range(len(self.uav_list)):
                state[(1 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0] +
                                             self.one_side_length, self.one_side_length] = 1

        if gp.OBSERVATION_VERSION == 7:
            for uav_index in range(len(self.uav_list)):
                temp = state[(1 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0]
                                            :(2 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0], :]
                temp[temp != 0] = 1
                state[(1 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0]
                                        :(2 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0], :] = temp
        if gp.OBSERVATION_VERSION == 8:
            for uav_index in range(len(self.uav_list)):
                temp = state[1 + uav_index * gp.OBSERVATION_DIMS]
                temp[temp != 0] = 1
                state[1 + uav_index * gp.OBSERVATION_DIMS] = temp
        return state

    def add_previous_action(self, action):
        # TODO: Change this part if the position of ap's postion layer changed in observation
        if 5 <= gp.OBSERVATION_VERSION < 7:
            for index, (aps_state, act) in enumerate(zip(self.state_buffer, action)):
                for uav_index in range(len(self.uav_list)):
                    if act != uav_index:
                        self.state_buffer[index][-1][(1 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0] +
                                                     self.one_side_length, self.one_side_length] = -1

        number_of_ap_per_edge = math.floor(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT + 1)
        if gp.OBSERVATION_VERSION == 7:
            for index, (aps_state, act) in enumerate(zip(self.state_buffer, action)):
                for uav_index in range(len(self.uav_list)):
                    temp = self.state_buffer[index][-1][(1 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0]
                                                        :(2 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0], :]
                    if act != uav_index:
                        temp[self.one_side_length, self.one_side_length] = -1
                    if (0 <= index - 1 < len(action)) and (index % number_of_ap_per_edge) != 0:
                        if action[index - 1] != uav_index:
                            temp[self.one_side_length, 0] = -1
                        else:
                            temp[self.one_side_length, 0] = 1
                        if 0 <= index - 1 - number_of_ap_per_edge < len(action):
                            if action[index -1 - number_of_ap_per_edge] != uav_index:
                                temp[0, 0] = -1
                            else:
                                temp[0, 0] = 1
                        if 0 <= index - 1 + number_of_ap_per_edge < len(action):
                            if action[index - 1 + number_of_ap_per_edge] != uav_index:
                                temp[self.one_side_length * 2, 0] = -1
                            else:
                                temp[self.one_side_length * 2, 0] = 1
                    if (0 <= index + 1 < len(action)) and ((index + 1) % number_of_ap_per_edge) != 0:
                        if action[index + 1] != uav_index:
                            temp[self.one_side_length, self.one_side_length * 2] = -1
                        else:
                            temp[self.one_side_length, self.one_side_length * 2] = 1
                        if 0 <= index + 1 + number_of_ap_per_edge < len(action):
                            if action[index + 1 + number_of_ap_per_edge] != uav_index:
                                temp[self.one_side_length * 2, self.one_side_length * 2] = -1
                            else:
                                temp[self.one_side_length * 2, self.one_side_length * 2] = 1
                        if 0 <= index + 1 - number_of_ap_per_edge < len(action):
                            if action[index + 1 - number_of_ap_per_edge] != uav_index:
                                temp[0, self.one_side_length * 2] = -1
                            else:
                                temp[0, self.one_side_length * 2] = 1
                    if 0 <= index + number_of_ap_per_edge < len(action):
                        if action[index + number_of_ap_per_edge] != uav_index:
                            temp[self.one_side_length * 2, self.one_side_length] = -1
                        else:
                            temp[self.one_side_length * 2, self.one_side_length] = 1
                    if 0 <= index - number_of_ap_per_edge < len(action):
                        if action[index - number_of_ap_per_edge] != uav_index:
                            temp[0, self.one_side_length] = -1
                        else:
                            temp[0, self.one_side_length] = 1
                    # self.state_buffer[1 + gp.OBSERVATION_DIMS * uav_index] = temp_state
                    # temp = self.rotate_single(temp, self.accesspoint_list[index])
                    self.state_buffer[index][-1][(1 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0]
                                                 :(2 + gp.OBSERVATION_DIMS * uav_index) * self.get_board_size()[0], :] = temp
        if gp.OBSERVATION_VERSION == 8:
            for index, (aps_state, act) in enumerate(zip(self.state_buffer, action)):
                for uav_index in range(len(self.uav_list)):
                    if act != uav_index:
                        self.state_buffer[index][-1][1 + uav_index * gp.OBSERVATION_DIMS][
                            self.one_side_length, self.one_side_length] = -1
            for index, (aps_state, act) in enumerate(zip(self.state_buffer, action)):
                for uav_index in range(len(self.uav_list)):
                    temp = self.state_buffer[index][-1][1 + uav_index * gp.OBSERVATION_DIMS]
                    if act != uav_index:
                        temp[self.one_side_length, self.one_side_length] = -1
                    if (0 <= index - 1 < len(action)) and (index % number_of_ap_per_edge) != 0:
                        if action[index - 1] != uav_index:
                            temp[self.one_side_length, 0] = -1
                        else:
                            temp[self.one_side_length, 0] = 1
                        if 0 <= index - 1 - number_of_ap_per_edge < len(action):
                            if action[index -1 - number_of_ap_per_edge] != uav_index:
                                temp[0, 0] = -1
                            else:
                                temp[0, 0] = 1
                        if 0 <= index - 1 + number_of_ap_per_edge < len(action):
                            if action[index - 1 + number_of_ap_per_edge] != uav_index:
                                temp[self.one_side_length * 2, 0] = -1
                            else:
                                temp[self.one_side_length * 2, 0] = 1
                    if (0 <= index + 1 < len(action)) and ((index + 1) % number_of_ap_per_edge) != 0:
                        if action[index + 1] != uav_index:
                            temp[self.one_side_length, self.one_side_length * 2] = -1
                        else:
                            temp[self.one_side_length, self.one_side_length * 2] = 1
                        if 0 <= index + 1 + number_of_ap_per_edge < len(action):
                            if action[index + 1 + number_of_ap_per_edge] != uav_index:
                                temp[self.one_side_length * 2, self.one_side_length * 2] = -1
                            else:
                                temp[self.one_side_length * 2, self.one_side_length * 2] = 1
                        if 0 <= index + 1 - number_of_ap_per_edge < len(action):
                            if action[index + 1 - number_of_ap_per_edge] != uav_index:
                                temp[0, self.one_side_length * 2] = -1
                            else:
                                temp[0, self.one_side_length * 2] = 1
                    if 0 <= index + number_of_ap_per_edge < len(action):
                        if action[index + number_of_ap_per_edge] != uav_index:
                            temp[self.one_side_length * 2, self.one_side_length] = -1
                        else:
                            temp[self.one_side_length * 2, self.one_side_length] = 1
                    if 0 <= index - number_of_ap_per_edge < len(action):
                        if action[index - number_of_ap_per_edge] != uav_index:
                            temp[0, self.one_side_length] = -1
                        else:
                            temp[0, self.one_side_length] = 1
                    # self.state_buffer[1 + gp.OBSERVATION_DIMS * uav_index] = temp_state
                    # temp = self.rotate_single(temp, self.accesspoint_list[index])
                    self.state_buffer[index][-1][1] = temp

    def convert_result_prob_to_popularity(self, result_prob):
        request_avaliable = self.scheduler_buffer[-1].numpy()
        request_avaliable = np.split(request_avaliable, gp.NUM_OF_UAV, axis=1)
        result = []
        for requests in request_avaliable:
            each_uav_request = np.zeros(int(self.get_resource_action_space()[0] / len(self.uav_list)))
            temp = np.split(requests, (int(np.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT))), axis=1)
            for each_column in temp:
                each_uav_request += \
                    np.sum(each_column.reshape((int(np.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT))), -1),
                           axis=0)
            result.append(each_uav_request)
        request_avaliable = np.array(result)
        # result = np.argsort(request_avaliable, axis=1)[:, -gp.DEFAULT_NUM_OF_RB * gp.DEFAULT_NUM_OF_RB_PER_RES:][:, ::-1]
        # uncomment for popularity_based scheduler
        request_avaliable[request_avaliable > 0] = 1
        each_prob = np.split(result_prob, len(self.uav_list))
        res_prob = np.multiply(request_avaliable, each_prob)
        result = np.argsort(res_prob, axis=1)[:, -gp.DEFAULT_NUM_OF_RB * gp.DEFAULT_NUM_OF_RB_PER_RES:][:, ::-1]
        if result.shape[0] != len(self.uav_list):
            raise ValueError("Dimension Not match")
        return result

    def form_popularity(self, result):
        index_list = []
        for index, array in enumerate(list(result)):
            array = array + index * gp.TOTAL_NUM_TILES
            index_list.extend(list(array))
        return index_list

    def obtain_action_from_popularity(self):
        action = [np.array([])] * len(self.uav_list)
        for _ in self.center_server.stack_of_popular:
            if self.center_server.stack_of_popular[_].__len__() == 0:
                action[_] = np.random.randint(0, high=self.uav_list[_].transmission_mask.size,
                                              size=gp.DEFAULT_NUM_OF_RB_PER_RES)
            if self.center_server.stack_of_popular[_].__len__() >= gp.DEFAULT_NUM_OF_RB_PER_RES:
                action[_] = np.array(self.center_server.stack_of_popular[_][-gp.DEFAULT_NUM_OF_RB_PER_RES::])
            else:
                action[_] = self.center_server.stack_of_popular[_][::]
                action[_] = np.concatenate(
                    (action[_], np.random.randint(0, high=self.uav_list[_].transmission_mask.size,
                                                  size=gp.DEFAULT_NUM_OF_RB_PER_RES - action[_].__len__())), axis=0)
        return action

    def accesspoint_tdma(self):
        action = [(self.center_server.clock % gp.DEFAULT_RESOURCE_BLOCKNUM) //
                  math.ceil(gp.DEFAULT_RESOURCE_BLOCKNUM / len(self.uav_list))] * len(self.accesspoint_list)
        return action

    def accesspoint_closest(self, ap_obs):
        action = []
        split = int(math.ceil(gp.ACCESS_POINTS_FIELD / gp.SQUARE_STEP)) * gp.OBSERVATION_DIMS
        for obs in ap_obs:
            x = torch.split(obs, split, dim=1)
            res = []
            for each in x:
                count = np.sum(each.numpy()[-1, -int(split/gp.OBSERVATION_DIMS)::, :])
                res.append(count)
            action.append(np.argmax(res))
        return action

    def step(self, controller, accesspoint, random=False, epsilon=0):
        """
            :parameter accesspoint: the models of access points
            :parameter accesspoint: the models of scheduler
            :parameter result_prob: output of network, with estimate weight of tiles for transmission
        """
        if random:
            sche_state = torch.stack(list(self.scheduler_buffer), dim=0)
            actual_action = self.convert_result_prob_to_popularity(controller)
            self.center_server.set_popularity_count(actual_action)
            self.form_popularity(actual_action)
            # self.center_server.centralized_clustering()
            # self.center_server.popularity_count_PF_clustered_channel_weighted()
            # actual_action = self.form_popularity(self.obtain_action_from_popularity())

            for index, tensor_obs in enumerate(self.get_observation_tensor()):
                self.state_buffer[index].append(tensor_obs)

            if self.center_server.clock == 0:
                for index, aps_reward in enumerate(self.reward_buffer):
                    self.original_obs.append(aps_reward[-1])
            ap_state = [torch.stack(list(aps_obv), dim=0) for aps_obv in self.state_buffer]

            # temp = np.array([torch.stack(list(aps_obv)[0::self.history_step_accesspoint], dim=0).numpy()
            #             for aps_obv in self.state_buffer])
            self.goto_next_state(accesspoint)
            self.add_previous_action(accesspoint)
            self.scheduler_buffer.append(self.obtain_popularity_count())

            return sche_state, ap_state, self.end_game()

        sche_state = torch.stack(list(self.scheduler_buffer), dim=0)
        if not controller.active:
            self.center_server.centralized_clustering()
            self.center_server.popularity_count_PF_clustered_channel_weighted()
            actual_action = self.form_popularity(self.obtain_action_from_popularity())
        else:
            actual_action = self.convert_result_prob_to_popularity(controller.act_e_greedy(
                sche_state, epsilon))
            self.center_server.set_popularity_count(actual_action)
            actual_action = self.form_popularity(actual_action)

        for index, tensor_obs in enumerate(self.get_observation_tensor()):
            self.state_buffer[index].append(tensor_obs)

        if self.center_server.clock == 0:
            for index, aps_reward in enumerate(self.reward_buffer):
                self.original_obs.append(aps_reward[-1])

        ap_state = [torch.stack(list(aps_obv), dim=0) for aps_obv in self.state_buffer]

        if not accesspoint[0].active:
            # action = self.accesspoint_tdma()
            action = self.accesspoint_closest(ap_state)
        else:
            action = []
            for index in range(len(self.accesspoint_list)):
                action.append(accesspoint[index].act_e_greedy(ap_state[index], self.available_ap[index],
                                                              epsilon, self.args.action_selection))
                # Choose an action greedily (with noisy weights)

        reward_sche, reward_ap, dec_actual_action = self.goto_next_state(action)

        if self.args.previous_action_observable:
            self.add_previous_action(dec_actual_action)
            ap_state = [torch.stack(list(aps_obv), dim=0) for aps_obv in self.state_buffer]

        self.scheduler_buffer.append(self.obtain_popularity_count())

        ap_return = (ap_state, dec_actual_action, [torch.tensor(dec_rew).to(device=self.args.device) for dec_rew in reward_ap])
        sche_return = (sche_state, actual_action, torch.tensor(reward_sche).to(device=self.args.device))
        return sche_return, ap_return, torch.tensor(self.end_game()).to(device=self.args.device)

    def step_p(self, controller, accesspoint):
        """
            :parameter accesspoint: the models of access points
            :parameter accesspoint: the models of scheduler
            :parameter result_prob: output of network, with estimate weight of tiles for transmission
        """
        sche_state = torch.stack(list(self.scheduler_buffer), dim=0)
        controller.send(sche_state)
        result_prob = controller.recv()
        if type(result_prob) is bool:
            self.center_server.centralized_clustering()
            self.center_server.popularity_count_PF_clustered_channel_weighted()
            actual_action = self.form_popularity(self.obtain_action_from_popularity())
        else:
            actual_action = self.convert_result_prob_to_popularity(result_prob)
            self.center_server.set_popularity_count(actual_action)
            actual_action = self.form_popularity(actual_action)

        for index, tensor_obs in enumerate(self.get_observation_tensor()):
            self.state_buffer[index].append(tensor_obs)

        if self.center_server.clock == 0:
            for index, aps_reward in enumerate(self.reward_buffer):
                self.original_obs.append(aps_reward[-1])

        ap_state = [torch.stack(list(aps_obv), dim=0) for aps_obv in self.state_buffer]

        action = []
        for index, pipe in enumerate(accesspoint):
            pipe.send((ap_state[index], self.available_ap[index]))
            action.append(pipe.recv())
        if action[0] is bool:
            # action = self.accesspoint_tdma()
            action = self.accesspoint_closest(ap_state)
                # Choose an action greedily (with noisy weights)

        reward_sche, reward_ap, dec_actual_action = self.goto_next_state(action)

        if self.args.previous_action_observable:
            self.add_previous_action(dec_actual_action)
            ap_state = [torch.stack(list(aps_obv), dim=0) for aps_obv in self.state_buffer]

        self.scheduler_buffer.append(self.obtain_popularity_count())

        ap_return = (ap_state, dec_actual_action, [torch.tensor(dec_rew).to(device=self.args.device) for dec_rew in reward_ap])
        sche_return = (sche_state, actual_action, torch.tensor(reward_sche).to(device=self.args.device))
        return sche_return, ap_return, torch.tensor(self.end_game()).to(device=self.args.device)

    def decentralized_reward(self):
        observation = [(np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)]))
                       for _ in range(len(self.uav_list))]
        psnr_list = self.center_server.count_instant_performance_linear()
        res_psnr = []
        for uav in self.uav_list:
            if uav.id in self.center_server.uav_user_match_table:
                res_psnr.append(self.plot_grid_map_linear_psnr_weighted(uav.id, psnr_list))
            else:
                res_psnr.append(observation[uav.id])

        pad_width = math.floor(1 + ((gp.ACCESS_POINTS_FIELD - 1) / 2 -
                                    (gp.LENGTH_OF_FIELD // gp.DENSE_OF_ACCESSPOINT) / 2) / gp.SQUARE_STEP)
        obs_decentral = []
        for index in range(len(self.uav_list)):
            obs_decentral.append(np.pad(res_psnr[index], int(pad_width), self.pad_with_zeros, padder=0))

        for ap_index, aps in enumerate(self.accesspoint_list):
            reward = []
            for index in range(len(self.uav_list)):
                a = math.floor(aps.position[0] / gp.SQUARE_STEP) + pad_width - self.one_side_length
                b = math.floor(aps.position[0] / gp.SQUARE_STEP) + pad_width + self.one_side_length + 1
                c = math.floor(aps.position[1] / gp.SQUARE_STEP) + pad_width - self.one_side_length
                d = math.floor(aps.position[1] / gp.SQUARE_STEP) + pad_width + self.one_side_length + 1
                temp_res = obs_decentral[index][int(a):int(b), int(c):int(d)]
                # temp_res = self.rotate_obs(temp_res, aps, dims=(0, 1))
                reward.append(temp_res)
            temp_reward = np.concatenate(reward, axis=1)
            temp_reward[self.reward_buffer[ap_index][-1] == 0] = 0
            if self.order_index:
                raise IndexError("Call reward function after getting new observation")
            self.reward_buffer[ap_index].append(temp_reward)
        # list ap: list uav: ndarray observation

        reward = []
        done = []
        frame_reward = []
        for ap_index in range(len(self.accesspoint_list)):
            if (self.center_server.clock + 1) % gp.DEFAULT_RESOURCE_BLOCKNUM == 0 and self.center_server.clock > 0:
                res_c = self.original_obs[ap_index] - self.reward_buffer[ap_index][-1]
                res_c = np.reshape(res_c[self.original_obs[ap_index].nonzero()], -1)
                if len(res_c) == 0:
                    frame_reward.append(0)
                else:
                    mean_res = res_c.mean()
                    var_res = res_c.var()
                    frame_reward.append(mean_res - var_res)

            res = self.reward_buffer[ap_index][1] - self.reward_buffer[ap_index][-1]
            done_ = np.sum(self.reward_buffer[ap_index][-1])
            mean_res = res.mean()
            var_res = res.var()
            reward.append((mean_res - var_res) * 10)  # reward scale up
            if done_ != 0:
                done_ = 1
            done.append(done_)
            # done is 0 if all users are fully satisfied
        if len(frame_reward) != 0:
            self.frame_reward.append(frame_reward)
        return np.array(reward), np.array(done)

    def obtain_centerlized_linear_reward(self):
        psnr = np.array(self.center_server.count_instant_performance_linear())
        res = psnr - self.centerlized_reward
        self.centerlized_reward = psnr
        mean_res = res.mean()
        var_res = res.var()
        return mean_res - var_res

    def decentra_finial_reward(self):
        reward = np.mean(np.array(self.frame_reward), axis=0)
        return reward

    def goto_next_state(self, action):
        if not self.order_index:
            raise AttributeError("Functions not in-order")
        # get observation should be called before this
        self.order_index = False

        dec_action = self.obtain_decision_from_action(action)
        reward, dec_finish_reward = self.center_server.perform_one_time_slot_transmission(np.array(dec_action))

        dec_reward, done_dec = self.decentralized_reward()  # regist old reward before new request comes

        # dec_reward[done_dec == 0.0] = 1
        dec_reward = dec_finish_reward - 0.5
        # reward = reward - 0.5
        # self.center_server.clock_tiktok()  # tiktok step by step
        self.center_server.clock_tiktok(not done_dec.any())  # if transmit finished jump to next one

        for ele in range(len(dec_reward)):
            if action[ele] != -1 and not done_dec[ele]:
                dec_reward[ele] = 1

        if not done_dec.any():
            # print("done", dec_action, self.center_server.clock)
            dec_reward = np.ones(dec_reward.size)
            # reward = np.ones(reward.size)
            # give finish reward

        self.center_server.connection_uav_ap.get_overall_fading()
        self.center_server.connection_ap_ue.get_overall_fading()
        return reward, dec_reward, dec_action

    def get_valid_action(self):
        """ never let action all false before end game """
        if not self.order_index:
            raise AttributeError("Functions not in-order")
        # get observation should be called before this
        action_mask = np.ones(self.action_size, dtype=bool)
        for uav in self.uav_list:
            if uav.id not in self.center_server.uav_user_match_table:
                action_mask = action_mask * self.action_masks[uav.id]
        if not action_mask.any():
            action_mask = np.ones(action_mask.shape, dtype=bool)
            # if all users in observations are served, just choose a uav to serve
        return action_mask

    def rot_pi(self, pi):
        """:return rotate pi indi*90 degree"""
        new_pi = np.zeros(len(pi))
        for key, nums in enumerate(pi):
            new_pi[self.pi_rot_bridge[key]] = nums
        return new_pi

    def get_streng_observations(self, pi):
        if not self.order_index:
            self.get_observation()
        result = [(self.observation, pi)]
        for indi in range(1, 4):
            # new_pi = self.rot_pi(new_pi)
            new_pi = pi
            result.append((np.rot90(self.observation, k=indi, axes=(0, 1)), new_pi))
            # data argumented by rotating
        return result

    def close(self):
        del self
        return
