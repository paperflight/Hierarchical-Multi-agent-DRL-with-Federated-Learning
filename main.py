import numpy as np
import Geometry as geo
import user_correlation as ucorr
import accesspoint as ap
import center_server as cs
import global_parameters as gp
from res_net.game import Centralized_Game
from res_net.coach import Coach
from res_net.basic_blocks import NNetwork as nnet

if __name__ == "__mainmain__":
    uav_list = []
    for i in range(0, gp.NUM_OF_UAV):
        uav_list.append(ucorr.UAV(i, np.array([np.random.randint(gp.LENGTH_OF_FIELD),
                                               np.random.randint(gp.WIDTH_OF_FIELD)]),
                                  [gp.LENGTH_OF_FIELD, gp.WIDTH_OF_FIELD], gp.UAV_FIELD_OF_VIEW))

    user_list = []
    for i in range(0, gp.NUM_OF_UAV):
        user_list.append(ucorr.User_VR(np.array([np.random.randint(gp.LENGTH_OF_FIELD),
                                                 np.random.randint(gp.WIDTH_OF_FIELD)]),
                                       [gp.LENGTH_OF_FIELD, gp.WIDTH_OF_FIELD], i,
                                       np.array([int(np.random.normal(0.5, 0.15) * gp.UAV_FIELD_OF_VIEW[0]),
                                                 int(np.random.normal(0.5, 0.15) * gp.UAV_FIELD_OF_VIEW[1])]),
                                       uav_list[np.random.randint(gp.NUM_OF_UAV)], gp.USER_FIELD_OF_VIEW))

    accesspoint_list = []
    for i in range(0, gp.LENGTH_OF_FIELD, gp.DENSE_OF_ACCESSPOINT):
        for j in range(0, gp.WIDTH_OF_FIELD, gp.DENSE_OF_ACCESSPOINT):
            accesspoint_list.append(ap.Access_Point(gp.LINK_THRESHOLD, int(i / gp.DENSE_OF_ACCESSPOINT) *
                                                    int(gp.WIDTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT)
                                                    + j / gp.DENSE_OF_ACCESSPOINT,
                                                    np.array([i, j])))

    center_server = cs.Center_Server(user_list, accesspoint_list, uav_list, gp.CORRELATION_THRESHOLD,
                                     gp.CLUSTERING_METHOD, gp.LINK_THRESHOLD)

    center_server.run_til_frame()

# first test program
if __name__ == "__main_main__":
    uav_list = []
    for i in range(0, 2):
        uav_list.append(ucorr.UAV(i, np.array([np.random.randint(50),
                                               np.random.randint(50)]),
                                  [50, 50], gp.UAV_FIELD_OF_VIEW))

    user_list = []
    for i in range(0, 70):
        position = np.array([np.random.randint(50), np.random.randint(50)])
        distance_to_uav = np.array([np.sum(np.power(position - np.array(uav.position), 2))
                                    for uav in uav_list])
        distance_to_uav = np.max(distance_to_uav) + 1 - distance_to_uav
        prob = distance_to_uav / np.sum(distance_to_uav)
        uav_id = np.random.choice(len(uav_list), p=prob)
        user_list.append(ucorr.User_VR(position,
                                       [50, 50], i,
                                       np.array([int(np.random.normal(0.5, 0.15) * gp.UAV_FIELD_OF_VIEW[0]),
                                                 int(np.random.normal(0.5, 0.15) * gp.UAV_FIELD_OF_VIEW[1])]),
                                       uav_list[uav_id], gp.USER_FIELD_OF_VIEW))

    accesspoint_list = []
    for i in range(0, 50, 25):
        for j in range(0, 50, 25):
            accesspoint_list.append(ap.Access_Point(gp.LINK_THRESHOLD, int(int(i / 25) *
                                                                           int(50 / 25) + j / 25),
                                                    np.array([i, j])))

    center_server = cs.Center_Server(user_list, accesspoint_list, uav_list, gp.CORRELATION_THRESHOLD,
                                     gp.CLUSTERING_METHOD, gp.LINK_THRESHOLD)

    print(center_server.run_til_frame())

if __name__ == "__test_main__":
    env = Centralized_Game()
    nnet = nnet(env)

    nnet.load_checkpoint(gp.LOAD_HISTORY_EXAMPLES_PATH[0])

    coach = Coach(env, nnet)
    coach.load_test(True)  # false to disable training and generate direct output

if __name__ == "__main__":
    env = Centralized_Game()
    nnet = nnet(env)

    if gp.LOAD_MODE:
        nnet.load_checkpoint(gp.LOAD_HISTORY_EXAMPLES_PATH[0])

    coach = Coach(env, nnet)
    if gp.LOAD_MODE:
        print("Load trainExamples from file")
        coach.load_train_replay()
    coach.learn()


# check the size of objects
def get_size(obj, seen=None):
    # From
    # Recursively finds size of objects
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
# Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
      size += sum([get_size(v, seen) for v in obj.values()])
      size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
      size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
      size += sum([get_size(i, seen) for i in obj])
    return size

if __name__ == "__xxmain__":
    import sys

    env = Centralized_Game()
    env.restart()
    print(get_size(env))