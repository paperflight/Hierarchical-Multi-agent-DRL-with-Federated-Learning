import numpy as np
import typing
import global_parameters as gp
import copy as cp


# not progressive raw video

class VR_Sphere:
    __slots__ = ['sphere_id', 'num_of_tile', 'tiles', 'transmission_mask', 'resource', 'its_resource']

    def __init__(self, sphere_id: int, tiles: list, num_of_tile: int, resources: np.ndarray):
        self.sphere_id: int = int(sphere_id)
        self.num_of_tile = num_of_tile
        self.tiles = tiles  # size of fov [x, y]
        self.transmission_mask = np.ones(self.num_of_tile, dtype=bool)
        if resources.size == 0:
            self.resource = np.arange(0, self.tiles[0] * self.tiles[1])
        else:
            if type(resources).__module__ != np.__name__:
                raise TypeError("Resource input should be numpy array")
            if self.num_of_tile != resources.__len__():
                raise ValueError("Size of resources not equal")
            if len(resources.shape) != 1:
                raise ValueError("Resource dimension should be 1")
            self.resource = resources
        self.its_resource = self.resource

    def __eq__(self, other):
        if self.sphere_id == other.sphere_id:
            if (np.equal(self.resource, other.resource)).all() and \
                    (np.equal(self.transmission_mask, other.transmission_mask)).all():
                return True
        return False

    def __sub__(self, other):
        if self.sphere_id != other.sphere_id:
            return 0
        return np.intersect1d(self.resource[self.transmission_mask],
                              other.resource[other.transmission_mask]).__len__()

    def __add__(self, other):
        if self.sphere_id != other.sphere_id:
            return 0
        return np.union1d(self.resource[self.transmission_mask],
                          other.resource[other.transmission_mask]).__len__()

    def __truediv__(self, other):
        if self.sphere_id != other.sphere_id:
            return self
        self.resource = np.setdiff1d(self.resource[self.transmission_mask],
                                     other.resource[other.transmission_mask])
        self.num_of_tile -= np.intersect1d(self.resource[self.transmission_mask],
                                           other.resource[other.transmission_mask]).__len__()

    def __mul__(self, other):
        if self.sphere_id != other.sphere_id:
            return self
        self.resource = np.union1d(self.resource[self.transmission_mask],
                                   other.resource[other.transmission_mask])
        self.num_of_tile = self.resource.__len__()

    def __mod__(self, other):
        self.sphere_id = other.sphere_id
        self.resource = other.resource
        self.num_of_tile = other.resource.__len__()
        self.transmission_mask = other.transmission_mask

    def __str__(self):
        return "VR Resource at " + str(self.sphere_id) + "\n Resources: " + str(self.resource[self.transmission_mask])


class Field_of_View(VR_Sphere):
    __slots__ = ['sphere_id', 'num_of_tile', 'tiles', 'transmission_mask', 'resource',
                 'center_resource', 'its_resource']

    def __init__(self, center_resource, resource_list, sphere_id, tiles):
        super(Field_of_View, self).__init__(sphere_id, tiles, resource_list.__len__(), resource_list)
        self.center_resource = center_resource

    def __mod__(self, other):
        super(Field_of_View, self).__mod__(other)
        self.center_resource = other.center_resource

    def __eq__(self, other):
        if super(Field_of_View, self).__eq__(other) and self.center_resource == other.center_resource:
            return True
        return False


class Clustering:
    __slots__ = ['correlation_matrix', 'cluster_method', 'cluster_num', 'cluster_threshold', 'cluster_result']

    def __init__(self, correlation_matrix, cluster_method=None, cluster_threshold=None, cluster_num=None):
        # correlation_matrix should be a two dim narray
        self.correlation_matrix = np.asarray(correlation_matrix)
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("correlation matrix should be square")
        if cluster_method is None or cluster_threshold is None:
            self.cluster_method = "k-mean"
            self.cluster_num = 4
            self.cluster_threshold = 0.05  # 1 correlation with 20 meters distance
        else:
            self.cluster_method = cluster_method
            self.cluster_num = cluster_num
            self.cluster_threshold = cluster_threshold
        self.cluster_method = cluster_method
        self.correlation_matrix = correlation_matrix
        for index in range(0, self.correlation_matrix.shape[0]):
            self.correlation_matrix[index][index] = 0
        self.cluster_result = []

    def cluster(self):
        self.correlation_matrix = np.where(self.correlation_matrix <=
                                           self.cluster_threshold, 0, self.correlation_matrix)
        max_clique = []
        temp_node_list = np.array([index for index in range(0, self.correlation_matrix.shape[0])
                                   if not (self.correlation_matrix[index] == 0).all()])
        clusted_user_list = cp.copy(temp_node_list)
        if self.cluster_method == "PrivotingBK":
            while temp_node_list.shape[0] != 0:
                result = []
                degency = self.degeneracy_ordering(temp_node_list)
                self.degeneracy_bk(degency, result)
                local_max_cli: np.ndarray = max(result, key=lambda p: p.shape[0])
                max_clique.append(local_max_cli)
                temp = local_max_cli
                result = [ele for ele in result if (np.intersect1d(ele, temp)).shape[0] == 0]
                while len(result) != 0:
                    local_max_cli = max(result, key=lambda p: p.shape[0])
                    temp = np.union1d(temp, local_max_cli)
                    result = [ele for ele in result if (np.intersect1d(ele, temp)).shape[0] == 0]
                    max_clique.append(local_max_cli)
                temp_node_list = np.setdiff1d(temp_node_list, temp)
            self.cluster_result = max_clique
        if self.cluster_method == "PrivotingBK_greedy":
            while temp_node_list.shape[0] != 0:
                result = self.greedy_bk(temp_node_list)
                temp_node_list = np.setdiff1d(temp_node_list, result)
                max_clique.append(result)
            self.cluster_result = max_clique
        if self.cluster_method == "k-mean":
            raise TypeError("Haven't implement k-mean!!")
        return clusted_user_list
        # Return node list which is being clustered, for additional clustering calculation.
        # in case of existing solo node which shows all 0 in correlation matrix
    # Clustering users

    def update_correlation_matrix(self, new_matrix):
        self.correlation_matrix = new_matrix

    def degeneracy_ordering(self, list_of_node):
        degeneracy = []
        degree = np.negative(np.ones(self.correlation_matrix.shape[0]))
        for ues in range(0, self.correlation_matrix.shape[0]):
            if ues in list_of_node:
                degree[ues] = np.sum(np.where(self.correlation_matrix[ues, :] != 0, 1, 0))
            else:
                degree[ues] = np.Inf
        for _ in list_of_node:
            minimum_degree = np.argmin(degree)
            if minimum_degree not in list_of_node:
                raise ValueError("Out of list range.")
            degree[np.nonzero(np.where(self.correlation_matrix[minimum_degree, :] != 0, 1, 0))] -= 1
            degeneracy.append(minimum_degree)
            degree[minimum_degree] = np.Inf
        return degeneracy

    def degeneracy_bk(self, degeneracy, result):
        p = degeneracy
        x = np.array([])
        for index, vertex in enumerate(degeneracy):
            neighbor = np.nonzero(np.where(self.correlation_matrix[vertex, :] != 0, 1, 0))
            r = np.array([vertex])
            self.privoting_bk(r, np.intersect1d(p, neighbor), np.intersect1d(x, neighbor), result)
            p = np.setdiff1d(p, vertex)
            x = np.union1d(x, vertex)

    def privoting_bk(self, r, p, x, result):
        if len(p) == 0 and len(x) == 0:
            result.append(r)
            return
        maximum_neighbor_num = 0
        maximum_neighbor = np.array([])
        for index in p:
            num_neighbor = np.sum(np.where(self.correlation_matrix[index, :] != 0, 1, 0))
            if num_neighbor > maximum_neighbor_num:
                maximum_neighbor_num = num_neighbor
                maximum_neighbor = np.nonzero(np.where(self.correlation_matrix[index, :] != 0, 1, 0))
        for vertex in np.setdiff1d(p, maximum_neighbor):
            self.privoting_bk(np.union1d(r, vertex),
                              np.intersect1d(p, np.nonzero(np.where(self.correlation_matrix[vertex, :] != 0, 1, 0))),
                              np.intersect1d(x, np.nonzero(np.where(self.correlation_matrix[vertex, :] != 0, 1, 0))),
                              result)
            p = np.setdiff1d(p, vertex)
            x = np.union1d(x, vertex)

    def greedy_bk(self, list_of_nodes):
        clique = np.array([], dtype=np.int)
        vertices = list_of_nodes
        rand = np.random.randint(len(vertices), size=1)
        clique = np.append(clique, vertices[rand])
        neighbor = []
        for index in range(0, self.correlation_matrix.shape[0]):
            neighbor.append(np.nonzero(np.where(self.correlation_matrix[index, :] != 0, 1, 0))[0])
        for v in vertices:
            if v in clique:
                continue
            is_next = True
            for u in clique:
                if u in neighbor[v]:
                    continue
                else:
                    is_next = False
                    break
            if is_next:
                clique = np.append(clique, v)
        return np.sort(clique)

    def get_cluster_result(self):
        return self.cluster_result


class User_VR(Field_of_View):
    __slots__ = ['sphere_id', 'num_of_tile', 'tiles', 'transmission_mask', 'resource', 'center_resource',
                 'id', 'position', 'mobility_range', 'original_source', 'sizeof_fov', 'resource_size', 'its_resource',
                 'in_range_ap']

    def __init__(self, position: np.ndarray, mobility_range, user_index, center_resource: np.ndarray,
                 original_source: VR_Sphere, ap_list, sizeof_fov=None, resource=None):
        # def __init__(self, center_resource, resource_list, sphere_id, tiles):
        # tiles: size of tiles [x,y], list
        # center_resource: postion of center tile: [x,y], list
        # resource_list: resource of list: numpy array (x,)
        self.id: int = int(user_index)
        self.position = position
        self.mobility_range = mobility_range
        self.original_source = original_source
        self.in_range_ap = []
        self.calculate_range(ap_list)

        if center_resource[0] * center_resource[1] >= original_source.num_of_tile:
            raise ValueError("No such resource")
        self.center_resource = center_resource

        if sizeof_fov is None:
            self.sizeof_fov = gp.USER_FIELD_OF_VIEW  # if no specific fov, use default fov 7x5 with 30x30 degree block
        elif sizeof_fov[0] % 2 == 0 or sizeof_fov[1] % 2 == 0:
            raise ValueError("FoV should be odd numbers")
        else:
            self.sizeof_fov = sizeof_fov

        if resource is None:
            resource_list = self.distribute_resource(original_source)
        else:
            resource_list = np.array(resource)

        super(User_VR, self).__init__(self.center_resource, resource_list, original_source.sphere_id, self.sizeof_fov)
        self.resource_size = np.ones(self.num_of_tile) * gp.TILE_SIZE

    # def __deepcopy__(self, memo):
    #     copied = User_VR(self.position, self.mobility_range, self.id, self.center_resource, self.original_source,
    #                      self.sizeof_fov, self.resource)
    #     copied.transmission_mask = self.transmission_mask.copy()
    #     copied.resource_size = np.copy(self.resource_size)
    #     return copied

    def clock_tiktok(self, gop):
        res_resource = self.resource[self.transmission_mask]
        res_resource_size = np.array([])
        if self.transmission_mask.any():
            res_resource_index = np.concatenate(np.argwhere(self.transmission_mask), axis=0)
            res_resource_size = self.resource_size[res_resource_index]
        res_resource += gp.TOTAL_NUM_TILES
        self.resource = np.concatenate((self.its_resource, res_resource))
        self.num_of_tile = len(self.resource)
        self.transmission_mask = np.ones(self.num_of_tile, dtype=bool)

        self.resource_size = np.concatenate((np.ones(len(self.its_resource)) * gp.GOP_TILE_SIZE[gop],
                                            res_resource_size))

    def get_resource_uav_id(self):
        return self.original_source.sphere_id

    def dist(self, other):
        return np.sqrt(np.power(self.position[0] - other.position[0], 2) +
                       np.power(self.position[1] - other.position[1], 2))

    def correlation(self, other):
        if self.sphere_id != other.sphere_id:
            return 0
        if not self.transmission_mask.any() and not other.transmission_mask.any():
            return 0
        if self.dist(other) == 0:
            return abs(self - other) / abs(self + other)
        return 1 / (self.dist(other)) * abs(self - other) / abs(self + other)

    def distribute_resource(self, original_source):
        resource = []

        width_list = []
        width_list_temp = list(range(0, original_source.tiles[1]))
        if self.center_resource[1] < int((self.sizeof_fov[1] - 1) / 2):
            width_list = list(width_list_temp[int(self.center_resource[1] - (self.sizeof_fov[1] - 1) / 2):])
            width_list.extend(width_list_temp[0:self.center_resource[1] + int((self.sizeof_fov[1] + 1) / 2)])
        elif int((self.sizeof_fov[1] - 1) / 2) <= self.center_resource[1] <= \
                int(original_source.tiles[1] - int((self.sizeof_fov[1] + 1) / 2)):
            width_list = [self.center_resource[1] + x for x in range(-int((self.sizeof_fov[1] - 1) / 2),
                                                                     int((self.sizeof_fov[1] + 1) / 2))]
        elif self.center_resource[1] > int(original_source.tiles[1] - int((self.sizeof_fov[1] + 1) / 2)):
            width_list = list(range(0, self.center_resource[1] + int((self.sizeof_fov[1] + 1) / 2 -
                                                                     original_source.tiles[1])))
            width_list.extend(list(range(self.center_resource[1] - int((self.sizeof_fov[1] - 1) / 2),
                                         original_source.tiles[1])))

        # central and edge region
        if int((self.sizeof_fov[0] - 1) / 2) <= self.center_resource[0] <= \
                int(original_source.tiles[0] - int((self.sizeof_fov[0] + 1) / 2)):
            height_list = [self.center_resource[0] + j for j in range(-int((self.sizeof_fov[0] - 1) / 2),
                                                                      int((self.sizeof_fov[0] + 1) / 2))]
            for m in width_list:
                for n in height_list:
                    resource.append(original_source.resource.reshape(original_source.tiles)[n][m])

        elif self.center_resource[0] < int((self.sizeof_fov[0] - 1) / 2):  # up polar region
            for m in range(0, int((self.sizeof_fov[0] - 1) / 2) - self.center_resource[0]):
                for n in range(0, original_source.tiles[1]):
                    resource.append(original_source.resource.reshape(original_source.tiles)[m][n])
            for m in range(int((self.sizeof_fov[0] - 1) / 2) - self.center_resource[0],
                           self.center_resource[0] + int((self.sizeof_fov[0] + 1) / 2)):
                for n in width_list:
                    resource.append(original_source.resource.reshape(original_source.tiles)[m][n])

        elif self.center_resource[0] > int(original_source.tiles[0] - int((self.sizeof_fov[0] + 1) / 2)):
            # down polar region
            for m in range(int(original_source.tiles[0] -((self.sizeof_fov[0] + 1) / 2 -
                           (original_source.tiles[0] - self.center_resource[0]))), original_source.tiles[0]):
                for n in range(0, original_source.tiles[1]):
                    resource.append(original_source.resource.reshape(original_source.tiles)[m][n])
            for m in range(int(self.center_resource[0] - (self.sizeof_fov[0] - 1) / 2),
                      int(original_source.tiles[0] - ((self.sizeof_fov[0] + 1) / 2 -
                                                      (original_source.tiles[0] - self.center_resource[0])))):
                for n in width_list:
                    resource.append(original_source.resource.reshape(original_source.tiles)[m][n])
        # temp = np.sort(np.array(resource))
        # show = np.ones(original_source.tiles[0] * original_source.tiles[1])
        # show[temp] = 0
        # show = np.reshape(show, original_source.tiles)
        return np.sort(np.array(resource))

    @staticmethod
    def limit_center_range(input_shape, shape_range):
        if input_shape[0] < 0:
            input_shape[0] = 0
        elif input_shape[0] >= shape_range[0]:
            input_shape[0] = shape_range[0] - 1
        if input_shape[1] < 0:
            input_shape[1] = 0
        elif input_shape[1] >= shape_range[1]:
            input_shape[1] = shape_range[1] - 1
        return input_shape

    # resize the center resource inside current resource range

    def moving_fov(self, delta, size_of_fov=None, new_source=None, new_center=None):
        # if entering new_source, inter field moving, else, intra moving
        if size_of_fov is not None:
            self.sizeof_fov = size_of_fov
        if new_source is None:
            self.center_resource[0] += delta[0]
            self.center_resource[1] += delta[1]
            self.center_resource = self.limit_center_range(self.sizeof_fov, self.original_source.tiles)
            new_fov_list = self.distribute_resource(self.original_source)
            new_fov = Field_of_View(self.center_resource, new_fov_list, self.original_source.sphere_id, self.sizeof_fov)
            self % new_fov
        else:
            self.center_resource = new_center
            self.original_source = new_source
            self.center_resource = self.limit_center_range(self.sizeof_fov, self.original_source.tiles)
            new_fov_list = self.distribute_resource(self.original_source)
            new_fov = Field_of_View(self.center_resource, new_fov_list, self.original_source.sphere_id, self.sizeof_fov)
            self % new_fov

    def calculate_range(self, ap_list):
        position = np.array([aps.position for aps in ap_list])
        compare_res = np.abs(position - np.array(self.position)) <= ((gp.ACCESS_POINTS_FIELD - 1) / 2 * gp.REWARD_CAL_RANGE)
        index = np.logical_and(compare_res[:, 0], compare_res[:, 1]).astype(np.bool)
        self.in_range_ap = [aps for aps in np.nonzero(index)[0].astype(int)]

    def mobility(self, delta, ap_list):
        self.position[0] += delta[0]
        self.position[1] += delta[1]
        self.calculate_range(ap_list)
        self.position = self.limit_center_range(self.position, self.mobility_range)

    def merge_into(self, count_dict: typing.Dict[int, int], incresing_step=1):
        # [resource id, count number]
        for key in enumerate(self.resource):
            if self.transmission_mask[key[0]] != 0:
                temp = key[1]
                if key[1] >= gp.TOTAL_NUM_TILES:
                    increasing = 0
                    while temp >= gp.TOTAL_NUM_TILES:
                        increasing += incresing_step
                        temp -= gp.TOTAL_NUM_TILES
                    count_dict[temp] += increasing
                    # this part of code is for PF scheduling with time +1
                count_dict[temp] += incresing_step
        return count_dict

    def transmitted(self, resource_id: np.ndarray, transmission_amount):
        amount = transmission_amount
        my_resource = np.in1d(self.resource % gp.TOTAL_NUM_TILES, resource_id)
        if my_resource.any():
            ind = np.max(np.where(my_resource)[0])
            if not self.transmission_mask[ind]:
                return True, False, False
            if amount >= np.sum(self.resource_size[ind]) / len(resource_id) * np.sum(my_resource):
                self.transmission_mask[ind] = 0
                return True, True, True
            else:
                return True, True, False
        return False, False, False  # tile exist? transmitted? sucessfull transmitted?
        # if requires that resource, receive, if not, wait and waste this turn

    def obtain_psnr(self):
        #  must call after the change of fov
        #  reset parameters
        temp_transmission_mask = self.transmission_mask[0:len(self.its_resource)].copy()
        for key, ele in enumerate(self.resource[len(self.its_resource)::]):
            if self.transmission_mask[key]:
                temp_transmission_mask[np.where(self.its_resource == ele % gp.TOTAL_NUM_TILES)[0]] = True
        if np.sum(temp_transmission_mask) == 0:
            current_psnr = 10 * np.log10(self.its_resource.size * 2)
        else:
            current_psnr = 10 * np.log10(1/(1/self.its_resource.size *
                                            (np.sum(temp_transmission_mask[0:len(self.its_resource)]))))
        return current_psnr

    def obtain_psnr_linear(self):
        #  must call after the change of fov
        #  reset parameters
        temp_transmission_mask = self.transmission_mask[0:len(self.its_resource)].copy()
        for key, ele in enumerate(self.resource[len(self.its_resource)::]):
            if self.transmission_mask[key]:
                temp_transmission_mask[np.where(self.its_resource == ele % gp.TOTAL_NUM_TILES)[0]] = True
        if np.sum(temp_transmission_mask) == 0:
            current_psnr = 1
        else:
            current_psnr = 1 - (1 / self.its_resource.size *
                                (np.sum(temp_transmission_mask[0:len(self.its_resource)])))
        return current_psnr


class UAV(VR_Sphere):
    __slots__ = ['sphere_id', 'num_of_tile', 'tiles', 'transmission_mask', 'resource', 'its_resource',
                 'id', 'mobility_range', 'position']

    def __init__(self, uav_index, position: np.ndarray, mobility_range, tiles: list):
        self.tiles = tiles
        self.num_of_tile = tiles[0] * tiles[1]
        self.position = position
        self.id: int = int(uav_index)
        self.mobility_range = mobility_range
        super(UAV, self).__init__(uav_index, self.tiles, self.num_of_tile, np.array([]))

    def __str__(self):
        return "UAV " + str(self.id) + " in " + str(self.position) + " with " + str(self.tiles)

    # def __deepcopy__(self, memo):
    #     copied = UAV(self.id, self.position, self.mobility_range, self.tiles)
    #     return copied

    @staticmethod
    def limit_center_range(input_shape, shape_range):
        if input_shape[0] < 0:
            input_shape[0] = 0
        elif input_shape[0] >= shape_range[0]:
            input_shape[0] = shape_range[0]
        if input_shape[1] < 0:
            input_shape[1] = 0
        elif input_shape[1] >= shape_range[1]:
            input_shape[1] = shape_range[1]
        return input_shape

    # resize the center resource inside current resource range

    def mobility(self, delta):
        self.position[0] += delta[0]
        self.position[1] += delta[1]
        self.position = self.limit_center_range(self.position, self.mobility_range)


if __name__ == "__correlation_test__":
    import time

    start_time = time.time()
    size = 10000
    N = np.random.rand(size, size)
    # for i in range (0, 100):
    #     for j in range (0, 100):
    #         if abs(i - j) > 20:
    #             N[i][j] = 0
    for i in range(0, size):
        N[i][i] = 0
    # for i in range (0, size):
    #     for j in range (0, size):
    #         if abs(i - j) > 200:
    #             N[i][j] = 0
    clst = Clustering(N, "PrivotingBK_greedy", 0.5)
    print(clst.get_cluster_result())
    print("--- %s seconds ---" % (time.time() - start_time))
