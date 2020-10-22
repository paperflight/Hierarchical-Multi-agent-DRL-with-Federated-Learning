import numpy as np

LENGTH_OF_FIELD = 80
WIDTH_OF_FIELD = 80  # must be diviable by square step
ACCESS_POINTS_FIELD = 61  # must be odd
REWARD_CAL_RANGE = 1  # reward calculation range for each accesspoint (range = RCR*ACCESS_FIELD)
DENSE_OF_ACCESSPOINT = 30
ACCESS_POINT_PER_EDGE = int(1 + LENGTH_OF_FIELD // DENSE_OF_ACCESSPOINT)

DENSE_OF_USERS = 120
VARIANCE_OF_USERS = 0

MAX_USERS_MOBILITY = 1

NUM_OF_UAV = 4
MAX_UAV_MOBILITY = 2

# PCP parameters
NUM_OF_CLUSTER = 1
DENSE_OF_USERS_PCP = int(DENSE_OF_USERS / NUM_OF_UAV / NUM_OF_CLUSTER)
UE_SCALE = 20
VARIANCE_OF_SCALE = 0
CLUSTER_SCALE = 1


def REFRESH_SCALE(ue_scale):
    global DENSE_OF_USERS, VARIANCE_OF_SCALE, VARIANCE_OF_USERS, DENSE_OF_USERS_PCP, UE_SCALE
    DENSE_OF_USERS = int((1 + ((np.random.rand() - 0.5) * VARIANCE_OF_USERS)) * DENSE_OF_USERS)
    DENSE_OF_USERS_PCP = int(DENSE_OF_USERS / NUM_OF_UAV / NUM_OF_CLUSTER)
    UE_SCALE = np.ceil((1 + ((np.random.rand() - 0.5) * VARIANCE_OF_SCALE)) * ue_scale)


FRAME_RATE = 90
TILE_SIZE: float = 30 * 30 * 8 * 60 * 3 / 150  # 8640
GOP: int = 5
GOP_TILE_SIZE: list = [TILE_SIZE, 0.7 * TILE_SIZE, 0.7 * TILE_SIZE,
                       0.7 * TILE_SIZE, 0.7 * TILE_SIZE, 0.7 * TILE_SIZE, 0.7 * TILE_SIZE]
GOP_INDEX: int = 0
GOP_SIZE_CONSTANT = False

UAV_FIELD_OF_VIEW = [6, 12]
TOTAL_NUM_TILES = UAV_FIELD_OF_VIEW[0] * UAV_FIELD_OF_VIEW[1]
USER_FIELD_OF_VIEW = [5, 7]  # 150 verti * 210 hori

LINK_THRESHOLD = 1e-7
CORRELATION_THRESHOLD = 0.00125

UAV_TRANSMISSION_CENTER_FREUENCY = 5e9
UAV_INTERFERENCE = False
AP_TRANSMISSION_CENTER_FREUENCY = 5e9

SPEED_OF_LIGHT = 3e8
DRONE_HEIGHT = 40
EXCESSIVE_NLOS_ATTENUATION = pow(10, 20 / 10)

ACCESS_POINT_TRANSMISSION_EIRP = pow(10, 78 / 10)  # 78 dBm
ACCESS_POINT_TRANSMISSION_BANDWIDTH = 50e6  # Hz
UAV_TRANSMISSION_EIRP = pow(10, 78 / 10)  # 78 dBm
UAV_TRANSMISSION_BANDWIDTH = 50e6  # Hz
NOISE_THETA = pow(10, -91 / 10)  # -91 hertz
UAV_AP_ALPHA = -2
AP_UE_ALPHA = -4
NAKAGAMI_M = 2
RAYLEIGH = 2
# https://arxiv.org/pdf/1704.02540.pdf

DEFAULT_RESOURCE_BLOCKNUM = 28  # NUM of blocks for association change
DEFAULT_RESOURCE_ALLOCATION = [1 / (FRAME_RATE * DEFAULT_RESOURCE_BLOCKNUM)
                               for ind in range(0, DEFAULT_RESOURCE_BLOCKNUM * GOP)]
DEFAULT_NUM_OF_RB_PER_RES = 10  # num of transmission tile in each decision slot
DEFAULT_NUM_OF_RB = 1  # num of transmission tile in each transmit slot
# transmit 5 slot, each with 10 tiles
# 60Hz --- 0.016666666667s
# into 20 slots

CLUSTERING_METHOD = "PrivotingBK_greedy"  # PrivotingBK_greedy/PrivotingBK

LOG_LEVEL = 0  # 0: nothing, 1: text_only, 2: rich text, 3: even detail+figure, 4: save figure
AP_COLOR = 'red'
UE_COLOR = 'green'
CS_COLOR = 'blue'

PLOT_FADING_RANGE_LOG = [-150., 0.]
IMAGE_PER_ROW = 4
IMAGE_SIZE = (5, 4)



# neural network parameters
LR = 1e-3
NUM_RES_BLOCKS = 8
NUM_CHANNELS = 128
DROP_OUT = 0.3
EPOCHS = 10
BATCH_SIZE = 64

# mcts parameters
MCTS_NUM = 25
C_PUCT = 1

# training parameters
# observation square step
SQUARE_STEP = 2
# larger than this number. perform pure greedy,
# should smaller than DEFAULT_RESOURCE_BLOCKNUM
# is the v_resign in the paper
ITERATION_NUM = 1000
HISTORY_MAX_LEN = 200000  # learning iteration numbers
EPS_NUM = 60  # episode in each iter
EPS_GREEDY_NUM = 55  # default policy starts from step xx, this should be smaller than maximum steps
TOTAL_HISTORY_LEN_IN_ITERATION = 50  # maximum learning history holded by deque
REPLAY_FILE_PATH = "./replay/"  # maximum length of total history in the number of iteration
UPDATE_THRESHOLD = 0.55  # Arena update threshold
ARENA_MATCH_NUM = 40  # Arena match num
LOAD_HISTORY_EXAMPLES_PATH = ('./temp', 'best.pth.tar')  # model and history load path
USER_CLUSTER_INDICATOR_LENGTH = 8
# indicate the number of clusters inside the figure 0 - 1 - step - step^2...
# if step=2 : 0, 1, 2, 4, 8
USER_CLUSTER_INDICATOR_STEP = 2  # scale the length indicator to reduce the states num
# TODO: Observation version 1-3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 4
OBSERVATION_DIMS = 3  # each cluster has three observations: uav position, user position, num of cluster
REWARD_STAGE = [10, 15, 20]  # reward stage, correspoinding to -1, 0, 1, 1.5
BIAS_SIGMA = 0.25  # bias sigma for ensure all moves may be tried
DIRICHLET = 0.03  # Bias parameter for dirichlet
# TODO: when selecting observation 4 and 5, change the observation dims too
OBSERVATION_VERSION = 7  # 1: observation v1, 2 observation v2
# for details look into game.get_observation_vx() function
NULL_UNAVALIABLE_UAV = False


# training and loading parameters
ENABLE_MODEL_RELOAD = False
ENABLE_MEMORY_RELOAD = False
ENABLE_EARLY_STOP = False
ENABLE_EARLY_STOP_THRESHOLD = 0.5
LOAD_MODE = False
PARALLEL_EXICUSION = True
ALLOCATED_CORES = 4

# What should notice when running a new job:
# OBSERVATION_VERSION : decide your observation type
# ALLOCATED_CORES : num of parallel cores based on your computer build
# LOAD_MODE: load previous model and playback in /temp/ or not
# EPS_GREEDY_NUM : MCTS search steps
# EPS_NUM : episodes for getting playback with new model
# ARENA_MATCH_NUM : the number of arena match for comparing two models
# TOTAL_HISTORY_LEN_IN_ITERATION : maximum stored history
