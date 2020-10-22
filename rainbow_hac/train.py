# -*- coding: utf-8 -*-
#from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import sys

sys.path.append('../..')
sys.path.append('./')

import pickle
import global_parameters as gp

import numpy as np
import math
import copy
import torch
from tqdm import trange

import multiprocessing
import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# TODO: When running in server, uncomment this line if needed
import copy as cp

from rainbow_hac.ap_agent import Agent
from rainbow_hac.center_agent import CT_Agent as Controller
from rainbow_hac.game import Decentralized_Game as Env
from rainbow_hac.memory import ReplayMemory
from rainbow_hac.test import test, test_p

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--active-scheduler', action='store_false', help='Active scheduler')
parser.add_argument('--active-accesspoint', action='store_false', help='Active AP')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
# parser.add_argument('--game', type=str, default='transmit-vr', choices=atari_py.list_games(), help='Environment game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length in game frames (0 to disable)')
# TODO: Note that the change of UAV numbers should also change the history-length variable
parser.add_argument('--previous-action-observable', action='store_true', help='Observe previous action? (AP)')
parser.add_argument('--history-length-accesspoint', type=int, default=2, metavar='T',
                    help='Total number of history state')
parser.add_argument('--history-length-scheduler', type=int, default=1, metavar='T',
                    help='Total number of history state')
parser.add_argument('--state-dims', type=int, default=gp.NUM_OF_UAV * gp.OBSERVATION_DIMS, metavar='S',
                    help='Total number of dims in consecutive states processed, UAV * 3 for current version')
parser.add_argument('--dense-of-uav', type=int, default=gp.NUM_OF_UAV, metavar='UAV',
                    help='Total number of UAVs')
parser.add_argument('--user-cluster-scale', type=int, default=gp.UE_SCALE, metavar='UAV',
                    help='Total number of UAVs')
parser.add_argument('--architecture', type=str, default='canonical_4uav_61obv_3x3_mix',
                    choices=['canonical_2uav_61obv_3x3_mix', 'canonical_4uav_61obv_3x3_mix',
                             'canonical_2uav_61obv_2x2_mix', 'canonical_4uav_61obv_2x2_mix',
                             'canonical_2uav_61obv_3x3', 'canonical_4uav_61obv_3x3',
                             'canonical_2uav_61obv_2x2', 'canonical_4uav_61obv_2x2',
                             'canonical_2uav_41obv_3x3_mix', 'canonical_4uav_41obv_3x3_mix',
                             'canonical_2uav_41obv_2x2_mix', 'canonical_4uav_41obv_2x2_mix',
                             'canonical_2uav_41obv_3x3', 'canonical_4uav_41obv_3x3',
                             'canonical_2uav_41obv_2x2', 'canonical_4uav_41obv_2x2',
                             'data-efficient', 'data-efficient_4uav_61obv_3x3_mix'],
                    metavar='ARCH', help='Network architecture')
# TODO: if select resnet8, obs v8 and dims 4 should be set in gp
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--noisy-std-controller-exploration', type=float, default=0.5, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--atoms-sche', type=int, default=21, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-2, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=2, metavar='V', help='Maximum of value distribution support')
# TODO: Make sure the value located inside V_min and V_max
parser.add_argument('--epsilon-min', type=float, default=0.0, metavar='ep_d', help='Minimum of epsilon')
parser.add_argument('--epsilon-max', type=float, default=0.0, metavar='ep_u', help='Maximum of epsilon')
parser.add_argument('--epsilon-delta', type=float, default=0.0001, metavar='ep_d', help='Decreasing step of epsilon')
# TODO: Set the ep carefully
parser.add_argument('--action-selection', type=str, default='greedy', metavar='action_type',
                    choices=['greedy', 'boltzmann', 'no_limit'],
                    help='Type of action selection algorithm, 1: greedy, 2: boltzmann')
parser.add_argument('--model', type=str, default=None, metavar='PARAM', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity-accesspoint', type=int, default=int(12e3), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--memory-capacity-scheduler', type=int, default=int(12e3), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--replay-frequency-scheduler', type=int, default=4, metavar='k',
                    help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step-accesspoint', type=int, default=3, metavar='n',
                    help='Number of steps for multi-step return')
parser.add_argument('--multi-step-scheduler', type=int, default=3, metavar='n',
                    help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.9, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8000), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--better-indicator', type=float, default=1.0, metavar='b',
                    help='The new model should be b times of old performance to be recorded')
# TODO: Switch interval should not be large
parser.add_argument('--learn-start', type=int, default=int(1000), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--learn-start-scheduler', type=int, default=int(1000), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--data-reinforce', action='store_true', help='DataReinforcement')
# TODO: Change this after debug
parser.add_argument('--evaluation-interval', type=int, default=500, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=20000, metavar='N',
                    help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
# TODO: Change this after debug
parser.add_argument('--evaluation-size', type=int, default=20, metavar='N',
                    help='Number of transitions to use for validating Q')
# TODO: This evaluation-size is used for Q value evaluation, can be small if Q is not important
parser.add_argument('--render', action='store_false', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0,
                    help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_false',
                    help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
# TODO: Change federated round each time
parser.add_argument('--federated-round', type=int, default=20, metavar='F',
                    help='Rounds to perform global combination, set a negative number to disable federated aggregation')

# Setup
args = parser.parse_args()

gp.UE_SCALE = args.user_cluster_scale

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('./results', args.id)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
# if torch.cuda.is_available() and not args.disable_cuda:
#     args.device = torch.device('cuda')
#     torch.cuda.manual_seed(np.random.randint(1, 10000))
#     torch.backends.cudnn.enabled = args.enable_cudnn
# else:
#     args.device = torch.device('cpu')
args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def average_weights(list_of_weight):
    """aggregate all weights"""
    averga_w = copy.deepcopy(list_of_weight[0])
    for key in averga_w.keys():
        for ind in range(1, len(list_of_weight)):
            averga_w[key] += list_of_weight[ind][key]
        averga_w[key] = torch.div(averga_w[key], len(list_of_weight))
    return averga_w


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip, scheduller_or_ap, index=-1):
    if not scheduller_or_ap:
        # save ap mem
        memory_path = memory_path[0:-4] + '_aps_' + str(index) + memory_path[-4:]
    else:
        memory_path = memory_path[0:-4] + '_sche' + memory_path[-4:]
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


def run_game_once_parallel_random(new_game, train_history_sche_parallel, train_history_aps_parallel, episode):
    train_examples_aps = []
    train_examples_sche = []
    for _ in range(number_of_aps):
        train_examples_aps.append([])
    eps, _, done_pp, _ = 0, None, True, None
    while eps < episode:
        if done_pp:
            _, done_pp = new_game.reset(), False

        sche_pack_p, aps_pack_p, done_pp = new_game.step(np.random.rand(scheduling_size[0]), [np.random.randint(0, action_space)
                                                                                           for _ in range(number_of_aps)], True)  # Step

        for index_p, ele_p in enumerate(aps_pack_p):
            train_examples_aps[index_p].append((ele_p, None, None, done_pp))

        train_examples_sche.append((sche_pack_p, None, None, done_pp))
        eps += 1
    train_history_aps_parallel.append(train_examples_aps)
    train_history_sche_parallel.append(train_examples_sche)


# Environment
env = Env(args)
env.reset()
action_space = env.get_action_size()
scheduling_size = env.get_resource_action_space()
number_of_aps = len(env.accesspoint_list)

# Controller
controller = Controller(args, env)

# Agent
dqn = []
matric = []
for _ in range(number_of_aps):
    # dqn.append(temp)
    dqn.append(Agent(args, env, _))
    matric.append(copy.deepcopy(metrics))

if args.federated_round > 0:
    global_model = Agent(args, env, "Global_")

# If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
    if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
    elif not os.path.exists(args.memory):
        raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

    mem_aps = []
    for index in range(number_of_aps):
        path = os.path.join(args.memory, ('metrics_aps' + str(index) + '.pth'))
        mem_aps.append(load_memory(path, args.disable_bzip_memory))
    path = os.path.join(args.memory, ('metrics_sche' + '.pth'))
    mem_sche = load_memory(path, args.disable_bzip_memory)

else:
    mem_aps = []
    for _ in range(number_of_aps):
        mem_aps.append(ReplayMemory(args, args.memory_capacity_accesspoint, True, env.remove_previous_action))
    mem_sche = ReplayMemory(args, args.memory_capacity_scheduler, False)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

# Construct validation memory
val_mem_aps = []
val_mem_sche = ReplayMemory(args, args.evaluation_size, False)
for _ in range(number_of_aps):
    val_mem_aps.append(ReplayMemory(args, args.evaluation_size, True, env.remove_previous_action))
if not gp.PARALLEL_EXICUSION:
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            _, done = env.reset(), False

        sche_pack, aps_pack, done = env.step(np.random.rand(scheduling_size[0]),
                                                    [np.random.randint(0, action_space)
                                                     for _ in range(number_of_aps)], True)
        val_mem_sche.append(sche_pack, None, None, done)
        for index, ele in enumerate(aps_pack):
            val_mem_aps[index].append(ele, None, None, done)
        T += 1
else:
    num_cores = min(multiprocessing.cpu_count(), gp.ALLOCATED_CORES) - 1
    num_eps = math.ceil(math.ceil(args.evaluation_size / num_cores) /
                        (gp.GOP * gp.DEFAULT_RESOURCE_BLOCKNUM)) * (gp.GOP * gp.DEFAULT_RESOURCE_BLOCKNUM)
    # make sure each subprocess can finish all the game (end with done)
    with multiprocessing.Manager() as manager:
        train_history_sche = manager.list()
        train_history_aps = manager.list()

        process_list = []
        for _ in range(num_cores):
            process = multiprocessing.Process(target=run_game_once_parallel_random,
                                              args=(cp.deepcopy(env), train_history_sche,
                                                    train_history_aps, num_eps))
            process_list.append(process)

        for pro in process_list:
            pro.start()
        for pro in process_list:
            pro.join()
            pro.terminate()

        for res in train_history_aps:
            for index, memerys in enumerate(res):
                for state, _, _, done in memerys:
                    val_mem_aps[index].append(state, None, None, done)
        for memorys in train_history_sche:
            for state, _, _, done in memorys:
                val_mem_sche.append(state, None, None, done)

if args.evaluate:
    controller.eval()
    for index in range(number_of_aps):
        dqn[index].eval()  # Set DQN (online network) to evaluation mode
    avg_reward, avg_Q = test(args, 0, controller, dqn, val_mem_aps, matric, val_mem_sche, metrics, results_dir, evaluate=True)  # Test
    for index in range(number_of_aps):
        print('Avg. reward for ap' + str(index) + ': ' + str(avg_reward[index]) + ' | Avg. Q: ' + str(avg_Q[index]))
else:
    # Training loop
    T, aps_state, done, sche_state, epsilon = 0, None, True, None, args.epsilon_max
    reinforce_ap = []
    for i in range(len(env.accesspoint_list)):
        temp = []
        for j in range(3):
            temp.append([])
        reinforce_ap.append(temp)
    reinforce_sche = []
    for i in range(3):
        reinforce_sche.append([])
    for T in trange(1, args.T_max + 1):
        # training loop
        if done:
            if T > 2:
                print(env.get_finial_reward())
            sche_state, done = env.reset(), False
            if T > 1 and args.data_reinforce:
                for sche_pair in reinforce_sche:
                    for sche_ele in sche_pair:
                        mem_sche.append(sche_ele[0], sche_ele[1], sche_ele[2], sche_ele[3])
                for index, ap_rein in enumerate(reinforce_ap):
                    for ap_pair in ap_rein:
                        for ap_ele in ap_pair:
                            mem_aps[index].append(ap_ele[0], ap_ele[1], ap_ele[2], ap_ele[3])
            reinforce_ap = []
            for i in range(len(env.accesspoint_list)):
                temp = []
                for j in range(3):
                    temp.append([])
                reinforce_ap.append(temp)
            reinforce_sche = []
            for i in range(3):
                reinforce_sche.append([])

        if T % args.replay_frequency == 0:
            controller.reset_noise()  # Draw a new set of noisy weights
            for _ in range(number_of_aps):
                dqn[_].reset_noise()

        sche_pack, aps_pack, done = env.step(controller, dqn, False, epsilon)  # Step
        epsilon = epsilon - args.epsilon_delta
        epsilon = np.clip(epsilon, a_min=args.epsilon_min, a_max=args.epsilon_max)

        if gp.ENABLE_EARLY_STOP:
            if env.center_server.clock % gp.DEFAULT_RESOURCE_BLOCKNUM == (gp.DEFAULT_RESOURCE_BLOCKNUM - 1):
                if env.center_server.obtain_centerlized_linear_reward() < gp.ENABLE_EARLY_STOP_THRESHOLD:
                    done = True

        reward_sche = sche_pack[2]
        if args.reward_clip > 0:
            reward_sche = torch.clamp(reward_sche, max=args.reward_clip, min=-args.reward_clip)  # Clip rewards
        mem_sche.append(sche_pack[0], sche_pack[1], reward_sche, done)  # Append transition to memory

        reward_aps = aps_pack[2]
        for _ in range(number_of_aps):
            if args.reward_clip > 0:
                reward_aps[_] = torch.clamp(reward_aps[_], max=args.reward_clip, min=-args.reward_clip) # Clip rewards
            if not aps_pack[1][_] == -1:
                mem_aps[_].append(aps_pack[0][_], aps_pack[1][_], reward_aps[_], done)  # Append transition to memory
            for direction in range(3):
                obs = aps_pack[0][_]
                if gp.OBSERVATION_VERSION <= 7:
                    res = []
                    rot_obs = torch.split(obs, int(obs.shape[1] / (gp.OBSERVATION_DIMS * gp.NUM_OF_UAV)), dim=1)
                    for index, ele in enumerate(rot_obs):
                        res.append(torch.rot90(ele, direction+1, (1, 2)))
                    obs = torch.cat(res, dim=1)
                if gp.OBSERVATION_VERSION == 8:
                    obs = torch.rot90(obs, direction, (2, 3))
                if not aps_pack[1][_] == -1:
                    reinforce_ap[_][direction].append((obs, aps_pack[1][_], reward_aps[_], done))
                # append rotated observation for data reinforcement
        obs = sche_pack[0][-1]
        rot_obs = list(torch.split(obs, gp.UAV_FIELD_OF_VIEW[1], dim=1))
        res = []
        for ele in rot_obs:
            if gp.GOP >= 2:
                res.append(torch.stack(torch.split(ele, gp.UAV_FIELD_OF_VIEW[0] * 2, dim=0)))
            else:
                res.append(torch.stack(torch.split(ele, gp.UAV_FIELD_OF_VIEW[0], dim=0)))
        res = torch.stack(res)
        res = torch.split(res, int(math.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT)), dim=0)
        for direction in range(3):
            result = []
            for ele in res:
                temp = torch.rot90(ele, direction+1, (0, 1)).reshape(
                    int(math.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT)), -1, gp.UAV_FIELD_OF_VIEW[1])
                result.append(torch.cat([temp[each, :, :] for each in range(temp.shape[0])], dim=1))
            result = torch.cat(result, dim=1).unsqueeze(0)
            reinforce_sche[direction].append((result, sche_pack[1], reward_sche, done))
            # append rotated observation for data reinforcement

        # Train and test
        if T >= args.learn_start_scheduler:
            mem_sche.priority_weight = min(mem_sche.priority_weight + priority_weight_increase, 1)
            # Anneal importance sampling weight β to 1

            if T % args.replay_frequency_scheduler == 0:
                controller.learn(mem_sche)  # Train with n-step distributional double-Q learning

            # If memory path provided, save it
            if args.memory is not None:
                save_memory(mem_sche, args.memory, args.disable_bzip_memory, True)

            # Update target network
            if T % args.target_update == 0:
                controller.update_target_net()

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                controller.save(results_dir, 'checkpoint_controller' + '.pth')

        if T >= args.learn_start:
            for index in range(number_of_aps):
                mem_aps[index].priority_weight = min(mem_aps[index].priority_weight + priority_weight_increase, 1)
                # Anneal importance sampling weight β to 1

            if T % args.replay_frequency == 0:
                for index in range(number_of_aps):
                    dqn[index].learn(mem_aps[index])  # Train with n-step distributional double-Q learning

            if T % args.federated_round == 0 and 0 < args.federated_round:
                global_weight = average_weights([model.get_state_dict() for model in dqn])
                global_model.set_state_dict(global_weight)
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' Global averaging starts')
                global_model.save(results_dir, 'Global_')
                for models in dqn:
                    models.set_state_dict(global_weight)

                # If memory path provided, save it
                for index in range(number_of_aps):
                    if args.memory is not None:
                        save_memory(mem_aps[index], args.memory, args.disable_bzip_memory, False, index)

                # Update target network
                if T % args.target_update == 0:
                    for index in range(number_of_aps):
                        dqn[index].update_target_net()

                # Checkpoint the network
                if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                    for index in range(number_of_aps):
                        dqn[index].save(results_dir, 'checkpoint' + str(index) + '.pth')

        if T % args.evaluation_interval == 0 and T > args.learn_start_scheduler and T > args.learn_start:
            controller.eval()  # Set DQN (online network) to evaluation mode
            for index in range(number_of_aps):
                dqn[index].eval()  # Set DQN (online network) to evaluation mode

            if gp.PARALLEL_EXICUSION:
                sche_pack, aps_pack = test_p(args, T, controller, dqn, val_mem_aps, matric, val_mem_sche,
                                             metrics, results_dir)  # Test
            else:
                sche_pack, aps_pack = test(args, T, controller, dqn, val_mem_aps, matric, val_mem_sche,
                                           metrics, results_dir)  # Test
            if sche_pack[2]:
                log('T = ' + str(T) + ' / ' + str(args.T_max) + '   Better model, accepted.')
            else:
                # mem_sche.expand_memory()
                log('T = ' + str(T) + ' / ' + str(args.T_max) + '   Worse model, reject.')
            log('T = ' + str(T) + ' / ' + str(args.T_max) + '  For controller'
                + ' | Avg. reward: ' + str(sche_pack[0]) + ' | Avg. Q: ' + str(sche_pack[1]))

            if aps_pack[2]:
                log('T = ' + str(T) + ' / ' + str(args.T_max) + '   Better model, accepted.')
            else:
                log('T = ' + str(T) + ' / ' + str(args.T_max) + '   Worse model, reject.')
            for index in range(number_of_aps):
                log('T = ' + str(T) + ' / ' + str(args.T_max) + '  For ap' + str(index) +
                    ' | Avg. reward: ' + str(aps_pack[0][index]) + ' | Avg. Q: ' + str(aps_pack[1][index]))

            controller.train()  # Set DQN (online network) back to training mode
            for index in range(number_of_aps):
                dqn[index].train()  # Set DQN (online network) back to training mode

env.close()
