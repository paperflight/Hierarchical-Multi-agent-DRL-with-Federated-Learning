# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
import multiprocessing
import copy as cp
import math
import global_parameters as gp

from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import numpy as np

from rainbow_hac.game import Decentralized_Game as Env


def test_parallel(new_game, c_pipe, c_pipe_controller, train_history_sche, train_history_aps, eps):
    train_examples_sche = []
    train_examples_aps = []
    for index in range(len(new_game.accesspoint_list)):
        train_examples_aps.append([])
    reward_sum_aps = []

    sche_state, done = None, True
    for _ in range(eps):
        while True:
            if done:
                sche_state, reward_sum_aps, reward_sum_sche, done = new_game.reset(), [], [], False

            sche_pack, aps_pack, done = new_game.step_p(c_pipe_controller, c_pipe)  # Step

            reward_sum_aps.append(aps_pack[2])
            if done:
                train_examples_sche.append(new_game.get_finial_reward())
                reward_sum_aps = np.mean(reward_sum_aps, axis=0)
                for index in range(len(new_game.accesspoint_list)):
                    train_examples_aps[index].append(reward_sum_aps[index])
                break
    train_history_sche.append(train_examples_sche)
    train_history_aps.append(train_examples_aps)

    for index in range(len(new_game.accesspoint_list)):
        c_pipe[index].send((np.array([False]), np.array([False])))
        c_pipe[index].close()
    c_pipe_controller.send(np.array([False]))
    c_pipe_controller.close()

    del new_game


# test whole system
def test(args, T, controller, dqn, val_mem_aps, metrics_aps, val_mem_sche, metrics_sche, results_dir, evaluate=False):
    env = Env(args)
    env.reset()

    T_rewards_sche, T_Qs_sche = [], []
    metrics_sche['steps'].append(T)
    T_rewards_aps, T_Qs_aps = [], []
    for _ in range(len(env.accesspoint_list)):
        metrics_aps[_]['steps'].append(T)
        T_rewards_aps.append([])
        T_Qs_aps.append([])

    # Test performance over several episodes
    sche_state, reward_sum, done = None, [], True
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                sche_state, reward_sum, done = env.reset(), [], False

            sche_pack, aps_pack, done = env.step(controller, dqn)

            reward_sum.append(aps_pack[2])
            if done:
                T_rewards_sche.append(env.get_finial_reward())
                reward_sum = np.mean(reward_sum, axis=0)
                print(np.mean(T_rewards_sche))
                for index in range(len(env.accesspoint_list)):
                    T_rewards_aps[index].append(reward_sum[index])
                break
    env.close()

    # Test Q-values over validation memory
    for state in val_mem_sche:  # Iterate over valid states
        T_Qs_sche.append(controller.evaluate_q(state))
    # Test Q-values over validation memory
    for index, val_mems in enumerate(val_mem_aps):
        for state in val_mems:  # Iterate over valid states
            T_Qs_aps[index].append(dqn[index].evaluate_q(state))

    avg_reward = sum(T_rewards_sche) / len(T_rewards_sche)
    avg_Q = sum(T_Qs_sche) / len(T_Qs_sche)

    avg_reward_aps, avg_Q_aps = [], []
    for _ in range(len(env.accesspoint_list)):
        avg_reward_aps.append(sum(T_rewards_aps[_]) / len(T_rewards_aps[_]))
        avg_Q_aps.append(sum(T_Qs_aps[_]) / len(T_Qs_aps[_]))

    better = True
    if not evaluate:
        # Save model parameters if improved
        if avg_reward > metrics_sche['best_avg_reward'] * args.better_indicator:
            metrics_sche['best_avg_reward'] = avg_reward
            controller.save(results_dir)
        # reload the state dict if obtain a better model

        # Append to results and save metrics
        metrics_sche['rewards'].append(T_rewards_sche)
        metrics_sche['Qs'].append(T_Qs_sche)
        torch.save(metrics_sche, os.path.join(results_dir, 'eval_metrics.pth'))

        # Plot
        _plot_line(metrics_sche['steps'], metrics_sche['rewards'], 'eval_Reward', path=results_dir)
        _plot_line(metrics_sche['steps'], metrics_sche['Qs'], 'eval_Q', path=results_dir)

    better_aps = True
    if not evaluate:
        # Save model parameters if improved
        better_vote = np.array([False] * len(env.accesspoint_list), dtype=np.int32)
        worse_vote = np.array([False] * len(env.accesspoint_list), dtype=np.int32)
        for _ in range(len(env.accesspoint_list)):
            if avg_reward_aps[_] > metrics_aps[_]['best_avg_reward'] * args.better_indicator:
                metrics_aps[_]['best_avg_reward'] = avg_reward_aps[_]
                dqn[_].save(results_dir, _)
                better_vote[_] = True
            elif avg_reward_aps[_] * args.better_indicator > metrics_aps[_]['best_avg_reward']:
                worse_vote[_] = True
        if np.sum(better_vote) >= np.ceil(len(env.accesspoint_list) / 3 * 2):
            if not np.sum(worse_vote) >= np.ceil(len(env.accesspoint_list) / 3 * 2):
                for _ in range(len(env.accesspoint_list)):
                    dqn[_].reload_step_state_dict()
        else:
            if gp.ENABLE_MODEL_RELOAD:
                for _ in range(len(env.accesspoint_list)):
                    dqn[_].reload_step_state_dict(False)
            better_aps = False
        # reload the state dict if obtain a better model

        for _ in range(len(env.accesspoint_list)):
            # Append to results and save metrics
            metrics_aps[_]['rewards'].append(T_rewards_aps[_])
            metrics_aps[_]['Qs'].append(T_Qs_aps[_])
            torch.save(metrics_aps[_], os.path.join(results_dir, 'metrics' + str(_) + '.pth'))

        for _ in range(len(env.accesspoint_list)):
            # Plot
            _plot_line(metrics_aps[_]['steps'], metrics_aps[_]['rewards'], 'Reward' + str(_), path=results_dir)
            _plot_line(metrics_aps[_]['steps'], metrics_aps[_]['Qs'], 'Q' + str(_), path=results_dir)

    # Return average reward and Q-value
    return (avg_reward, avg_Q, better), (avg_reward_aps, avg_Q_aps, better_aps)


# Test DQN
def test_p(args, T, controller, dqn, val_mem_aps, metrics_aps, val_mem_sche, metrics_sche, results_dir, evaluate=False):
    env = Env(args)
    env.reset()
    T_rewards_sche, T_Qs_sche = [], []
    metrics_sche['steps'].append(T)
    T_rewards_aps, T_Qs_aps = [], []
    for _ in range(len(env.accesspoint_list)):
        metrics_aps[_]['steps'].append(T)
        T_rewards_aps.append([])
        T_Qs_aps.append([])

    num_cores = math.floor(min(multiprocessing.cpu_count(), gp.ALLOCATED_CORES) - 1)
    num_eps = math.ceil(math.ceil(args.evaluation_episodes / num_cores) /
                        (gp.GOP * gp.DEFAULT_RESOURCE_BLOCKNUM))
    # make sure each subprocess can finish all the game (end with done)
    with multiprocessing.Manager() as manager:
        train_history_sche = manager.list()
        train_history_aps = manager.list()

        p_pipe_list1 = []
        c_pipe_list1 = []
        for _ in range(num_cores):
            p_pipe, c_pipe = multiprocessing.Pipe()
            p_pipe_list1.append(p_pipe)
            c_pipe_list1.append(c_pipe)

        p_pipe_list2 = []
        c_pipe_list2 = []
        for _ in range(num_cores):
            temp1, temp2 = [], []
            for temp in range(len(env.accesspoint_list)):
                p_pipe, c_pipe = multiprocessing.Pipe()
                temp1.append(p_pipe)
                temp2.append(c_pipe)
            p_pipe_list2.append(temp1)
            c_pipe_list2.append(temp2)
        p_pipe_list2 = np.array(p_pipe_list2)

        process_list = []
        for _ in range(num_cores):
            process = multiprocessing.Process(target=test_parallel,
                                              args=(cp.deepcopy(env), c_pipe_list2[_], c_pipe_list1[_],
                                                    train_history_sche, train_history_aps, num_eps))
            process_list.append(process)

        for pro in process_list:
            pro.start()

        on_off1 = True
        on_off2 = True
        while on_off1 or on_off2:
            on_off1 = controller.lookup_server_loop(p_pipe_list1)
            temp = np.ones(len(env.accesspoint_list), dtype=bool)
            for index in range(len(env.accesspoint_list)):
                temp[index] = dqn[index].lookup_server_loop(p_pipe_list2[:, index])
            on_off2 = temp.any()

        for pro in process_list:
            pro.join()
            pro.terminate()

        for res in train_history_sche:
            for reward in res:
                T_rewards_sche.append(reward)
        for res in train_history_aps:
            for index, memerys in enumerate(res):
                for reward in memerys:
                    T_rewards_aps[index].append(reward)

    # Test Q-values over validation memory
    for state in val_mem_sche:  # Iterate over valid states
        T_Qs_sche.append(controller.evaluate_q(state))
    # Test Q-values over validation memory
    for index, val_mems in enumerate(val_mem_aps):
        for state in val_mems:  # Iterate over valid states
            T_Qs_aps[index].append(dqn[index].evaluate_q(state))

    avg_reward_sche = sum(T_rewards_sche) / len(T_rewards_sche)
    avg_Q_sche = sum(T_Qs_sche) / len(T_Qs_sche)

    avg_reward_aps, avg_Q_aps = [], []
    for _ in range(len(env.accesspoint_list)):
        avg_reward_aps.append(sum(T_rewards_aps[_]) / len(T_rewards_aps[_]))
        avg_Q_aps.append(sum(T_Qs_aps[_]) / len(T_Qs_aps[_]))

    better_sche = True
    if not evaluate:
        # Save model parameters if improved
        if avg_reward_sche >= metrics_sche['best_avg_reward'] * args.better_indicator:
            if avg_reward_sche > metrics_sche['best_avg_reward']:
                metrics_sche['best_avg_reward'] = avg_reward_sche
            controller.save(results_dir)
            controller.reload_step_state_dict()
        else:
            if gp.ENABLE_MODEL_RELOAD:
                controller.reload_step_state_dict(False)
            better_sche = False
        # reload the state dict if obtain a better model

        # Append to results and save metrics
        metrics_sche['rewards'].append(T_rewards_sche)
        metrics_sche['Qs'].append(T_Qs_sche)
        torch.save(metrics_sche, os.path.join(results_dir, 'scheduler_metrics.pth'))

        # Plot
        _plot_line(metrics_sche['steps'], metrics_sche['rewards'], 'scheduler_Reward', path=results_dir)
        _plot_line(metrics_sche['steps'], metrics_sche['Qs'], 'scheduler_Q', path=results_dir)

    better_aps = True
    if not evaluate:
        # Save model parameters if improved
        better_vote = np.array([False] * len(env.accesspoint_list), dtype=np.int32)
        worse_vote = np.array([False] * len(env.accesspoint_list), dtype=np.int32)
        for _ in range(len(env.accesspoint_list)):
            if avg_reward_aps[_] >= metrics_aps[_]['best_avg_reward'] * args.better_indicator or better_sche:
                if avg_reward_aps[_] > metrics_aps[_]['best_avg_reward']:
                    metrics_aps[_]['best_avg_reward'] = avg_reward_aps[_]
                dqn[_].save(results_dir, _)
                better_vote[_] = True
            elif avg_reward_aps[_] * args.better_indicator > metrics_aps[_]['best_avg_reward'] or better_sche:
                worse_vote[_] = True
        if np.sum(better_vote) >= np.ceil(len(env.accesspoint_list) / 3 * 2):
            if not np.sum(worse_vote) >= np.ceil(len(env.accesspoint_list) / 3 * 2):
                for _ in range(len(env.accesspoint_list)):
                    dqn[_].reload_step_state_dict()
        else:
            if gp.ENABLE_MODEL_RELOAD:
                for _ in range(len(env.accesspoint_list)):
                    dqn[_].reload_step_state_dict(False)
            better_aps = False
        # reload the state dict if obtain a better model

        for _ in range(len(env.accesspoint_list)):
            # Append to results and save metrics
            metrics_aps[_]['rewards'].append(T_rewards_aps[_])
            metrics_aps[_]['Qs'].append(T_Qs_aps[_])
            torch.save(metrics_aps[_], os.path.join(results_dir, 'metrics' + str(_) + '.pth'))

        for _ in range(len(env.accesspoint_list)):
            # Plot
            _plot_line(metrics_aps[_]['steps'], metrics_aps[_]['rewards'], 'Reward' + str(_), path=results_dir)
            _plot_line(metrics_aps[_]['steps'], metrics_aps[_]['Qs'], 'Q' + str(_), path=results_dir)

    # Return average reward and Q-value
    return (avg_reward_sche, avg_Q_sche, better_sche), (avg_reward_aps, avg_Q_aps, better_aps)


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(
        1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour),
                         name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent),
                          name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)
