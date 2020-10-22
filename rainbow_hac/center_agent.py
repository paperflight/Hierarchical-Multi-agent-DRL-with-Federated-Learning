# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from rainbow_hac.basic_block_center import DQN
from rainbow_hac.basic_block_center_mix import DQN as DQN_M
import global_parameters as gp


class CT_Agent:
    def __init__(self, args, env):
        self.active = args.active_scheduler
        if not self.active:
            return
        self.env = env
        self.action_space, self.square_resource = env.get_resource_action_space()
        self.uav_num = len(env.uav_list)
        self.atoms = args.atoms_sche
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step_scheduler
        self.discount = args.discount
        self.device = args.device
        self.net_type = args.architecture

        if 'mix' in self.net_type:
            self.online_net = DQN_M(args, self.action_space).to(device=args.device)
        else:
            self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            self.model_path = os.path.join(args.model, "scheduler_model.pth")
            if os.path.isfile(self.model_path):
                state_dict = torch.load(self.model_path,
                                        map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'),
                                             ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'),
                                             ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                        del state_dict[old_key]  # Delete old keys for strict load_state_dict
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + self.model_path)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(self.model_path)

        self.online_net.train()

        if 'mix' in self.net_type:
            self.target_net = DQN_M(args, self.action_space).to(device=args.device)
        else:
            self.target_net = DQN(args, self.action_space).to(device=args.device)

        self.online_dict = self.online_net.state_dict()
        self.target_dict = self.target_net.state_dict()

        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    def reload_step_state_dict(self, better=True):
        if not self.active:
            return

        if better:
            self.online_dict = self.online_net.state_dict()
            self.target_dict = self.target_net.state_dict()
        else:
            self.online_net.load_state_dict(self.online_dict)
            self.target_net.load_state_dict(self.target_dict)

    def get_state_dict(self):
        if not self.active:
            return

        return self.online_net.state_dict()

    def set_state_dict(self, new_state_dict):
        if not self.active:
            return

        self.online_net.load_state_dict(new_state_dict)
        return

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        if not self.active:
            return

        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        if not self.active:
            return
        with torch.no_grad():
            prob = (self.online_net(state.unsqueeze(0)) * self.support).sum(2)
        return prob.numpy()[0]

    def lookup_server(self, list_of_pipe):
        if not self.active:
            return
        num_pro = len(list_of_pipe)
        list_pro = np.ones(num_pro, dtype=bool)
        with torch.no_grad():
            while list_pro.any():
                for key, pipes in enumerate(list_of_pipe):
                    if not pipes.closed and pipes.readable:
                        obs = pipes.recv()
                        if len(obs) == 1:
                            if not obs[0]:
                                pipes.close()
                                list_pro[key] = False
                                continue
                        if not self.active:
                            pipes.send(False)
                        else:
                            pipes.send(self.act(obs).numpy())
                        # convert back to numpy or cpu-tensor, or it will cause error since cuda try to run in
                        # another thread. Keep the gpu resource inside main thread

    def lookup_server_loop(self, list_of_pipe):
        num_pro = len(list_of_pipe)
        list_pro = np.ones(num_pro, dtype=bool)
        for key, pipes in enumerate(list_of_pipe):
            if not pipes.closed and pipes.readable:
                if pipes.poll():
                    obs = pipes.recv()
                    if type(obs) is np.ndarray:
                        pipes.close()
                        list_pro[key] = False
                        continue
                    if not self.active:
                        pipes.send(False)
                    else:
                        pipes.send(self.act(obs))
            else:
                list_pro[key] = False
            # convert back to numpy or cpu-tensor, or it will cause error since cuda try to run in
            # another thread. Keep the gpu resource inside main thread
        return list_pro.any()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.3):  # High ε can reduce evaluation scores drastically
        if not self.active:
            return
        return np.random.rand(self.action_space) if np.random.random() < epsilon else self.act(state)

    def convert_result_prob_to_popularity(self, result_prob, state):
        request_avaliable = state[-1].numpy()
        request_avaliable = np.split(request_avaliable, gp.NUM_OF_UAV, axis=1)
        result = []
        for requests in request_avaliable:
            each_uav_request = np.zeros(int(self.action_space / self.uav_num))
            temp = np.split(requests, (int(np.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT))), axis=1)
            for each_column in temp:
                each_uav_request += \
                    np.sum(each_column.reshape((int(np.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT))), -1), axis=0)
            result.append(each_uav_request)
        request_avaliable = np.array(result)
        request_avaliable[request_avaliable > 0] = 1
        each_prob = np.split(result_prob, self.uav_num)
        res_prob = np.multiply(request_avaliable, each_prob)
        result = np.argsort(res_prob, axis=1)[:, -gp.DEFAULT_NUM_OF_RB * gp.DEFAULT_NUM_OF_RB_PER_RES:][:, ::-1]
        if result.shape[0] != self.uav_num:
            raise ValueError("Dimension Not match")

        index_list = []
        for index, array in enumerate(list(result)):
            array = array + index * int(self.action_space / self.uav_num)
            index_list.extend(list(array))
        return index_list

    def learn(self, mem):
        if not self.active:
            return
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
        # for index in range(returns.shape[1]):
        # index = np.random.randint(0, returns.shape[1])
        # actions_temp = actions[:, index:returns.shape[1] * self.uav_num:returns.shape[1]]
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[[[x] for x in range(self.batch_size)], actions, :]  # log p(s_t, a_t; θonline)
        log_ps_a = torch.reshape(log_ps_a, (self.batch_size, returns.shape[1], -1, self.atoms)).mean(dim=2)
        # log_ps_a = torch.mean(log_ps_a, 1)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            dns = dns.sum(2)
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = torch.tensor([self.convert_result_prob_to_popularity(temp.numpy(), state)
                                              for temp, state in zip(dns, next_states)],
                                             dtype=torch.int64, device=self.device)
            self.target_net.reset_noise()  # Sample new target net noise
            # argmax_indices_ns = argmax_indices_ns[:, index:returns.shape[1] * self.uav_num:returns.shape[1]]
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[[[x] for x in range(self.batch_size)], actions, :]
            pns_a = torch.reshape(pns_a, (self.batch_size, returns.shape[1], -1, self.atoms)).mean(dim=2)
            # pns_a = torch.mean(pns_a, 1)
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(2) + nonterminals.unsqueeze(2) * (self.discount ** self.n) * (self.support.unsqueeze(0)).unsqueeze(0)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, 1, self.atoms, dtype=torch.float32)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions).unsqueeze(1)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
            # m = m.unsqueeze_(1).expand(self.batch_size, select_length, self.atoms)

        loss = -torch.sum(m * log_ps_a, 2).mean(dim=1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())
        # Update priorities of sampled transitions

    def learn_single(self, mem):
        if not self.active:
            return
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
        total_loss = 0
        # for index in range(returns.shape[1]):
        index = np.random.randint(0, returns.shape[1])
        actions_temp = actions[:, index:returns.shape[1] * self.uav_num:returns.shape[1]]
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[[[x] for x in range(self.batch_size)], actions_temp, :]  # log p(s_t, a_t; θonline)
        log_ps_a = torch.mean(log_ps_a, 1)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            dns = dns.sum(2)
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = torch.tensor([self.convert_result_prob_to_popularity(temp.numpy(), state)
                                              for temp, state in zip(dns, next_states)],
                                             dtype=torch.int64, device=self.device)
            self.target_net.reset_noise()  # Sample new target net noise
            argmax_indices_ns = argmax_indices_ns[:, index:returns.shape[1] * self.uav_num:returns.shape[1]]
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[[[x] for x in range(self.batch_size)], argmax_indices_ns, :]
            pns_a = torch.mean(pns_a, 1)
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns[:, index].unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(
                0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms, dtype=torch.float32)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
            # m = m.unsqueeze_(1).expand(self.batch_size, select_length, self.atoms)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        clip_grad_norm_(self.online_net.parameters(), 1.0, norm_type=1)
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())
        # Update priorities of sampled transitions

    def update_target_net(self):
        if not self.active:
            return
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='scheduler_model.pth'):
        if not self.active:
            return
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        if not self.active:
            return 0
        with torch.no_grad():
            dns = (self.online_net(state.unsqueeze(0)) * self.support).sum(2)[0]
            index = self.convert_result_prob_to_popularity(dns.numpy(), state)
            return torch.mean(dns[index]).item()

    def train(self):
        if not self.active:
            return
        self.online_net.train()

    def eval(self):
        if not self.active:
            return
        self.online_net.eval()
