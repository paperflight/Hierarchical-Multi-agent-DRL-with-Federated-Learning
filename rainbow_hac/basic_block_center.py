# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import global_parameters as gp


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, inputs):
        if self.training:
            return F.linear(inputs, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inputs, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.atoms = args.atoms_sche
        self.action_space = action_space
        self.archit = args.architecture

        if gp.GOP >= 2:
            self.split = gp.UAV_FIELD_OF_VIEW[0] * 2 * math.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT)
        else:
            self.split = gp.UAV_FIELD_OF_VIEW[0] * math.ceil(gp.LENGTH_OF_FIELD / gp.DENSE_OF_ACCESSPOINT)

        if 'canonical' in args.architecture and '2x2' in args.architecture:
            # self.convs = nn.Sequential(nn.Conv2d(args.history_length_scheduler, 32, 5, stride=2, padding=2, dilation=2), nn.LeakyReLU(),
            #                            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.LeakyReLU(),
            #                            nn.Conv2d(64, 128, 3, stride=1, padding=0), nn.LeakyReLU())
            #                            # nn.Conv2d(128, 256, 3, stride=1, padding=0), nn.LeakyReLU())
            # self.conv_output_size = 5120
            self.convs = nn.Sequential(nn.Conv2d(args.history_length_scheduler, 16, 5, stride=2, padding=4, dilation=2),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(16, 32, 3, stride=1, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 32, 3, stride=1, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Dropout(0.2))
            self.conv_output_size = 8192
        elif 'canonical' in args.architecture and '3x3' in args.architecture:
            self.convs = nn.Sequential(nn.Conv2d(args.history_length_scheduler, 16, 5, stride=2, padding=2, dilation=2),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(16, 32, 4, stride=1, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 32, 3, stride=1, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Conv2d(64, 64, 3, stride=2, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Dropout(0.2))
            self.conv_output_size = 1024  # 3x3 2: 3328  4: 7936
            # TODO: adding UAV requires pooling to reduce the number of parameters
        elif args.architecture == 'canonical_3d':
            self.convs = nn.Sequential(nn.Conv3d(1, 32, (2, 5, 5), stride=1, padding=4, dilation=2), nn.LeakyReLU(),
                                       nn.Conv3d(32, 64, (2, 3, 3), stride=1, padding=0), nn.LeakyReLU(),
                                       nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=0), nn.LeakyReLU(),
                                       nn.Conv3d(64, 128, (1, 3, 3), stride=1, padding=0), nn.LeakyReLU(),
                                       nn.Conv3d(128, 128, (1, 3, 3), stride=1, padding=0), nn.LeakyReLU(),
                                       nn.Dropout(0.2))
            self.conv_output_size = 8192
        elif args.architecture == 'data-efficient':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                       nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
            self.conv_output_size = 576
        else:
            raise TypeError('No such strucure')
        # TODO: Calculate the output_size carefully!!!
        # if args.architecture == 'canonical':
        #     self.convs = nn.Sequential(nn.Conv2d(args.state_dims, 32, 3, stride=1, padding=1), nn.ReLU(),
        #                                nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
        #                                nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
        #     self.conv_output_size = 576
        # elif args.architecture == 'data-efficient':
        #     self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 3, stride=1, padding=0), nn.ReLU(),
        #                                nn.Conv2d(32, 64, 3, stride=1, padding=0), nn.ReLU())
        #     self.conv_output_size = 576

        self.fc = nn.Sequential(nn.Linear(self.conv_output_size * gp.NUM_OF_UAV, args.dense_of_uav * args.hidden_size),
                                nn.Dropout(0.2), nn.LeakyReLU(),
                                nn.Linear(self.conv_output_size * gp.NUM_OF_UAV, args.dense_of_uav * args.hidden_size),
                                nn.Dropout(0.2), nn.LeakyReLU())

        self.fc_h_v = NoisyLinear(args.dense_of_uav * args.hidden_size, args.dense_of_uav * args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(args.dense_of_uav * args.hidden_size, args.dense_of_uav * args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.dense_of_uav * args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.dense_of_uav * args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        # if x.shape[1] != 1:
        #     list_x = torch.split(x, 1, dim=1)
        #     x = torch.cat(list_x, dim=2)
        x = torch.split(x, self.split, dim=3)
        res = []
        for index, each in enumerate(x):
            each = self.convs(each.float())
            res.append(each.view(each.size(0), -1))
        x = self.fc(torch.cat(res, dim=1))

        v = self.fc_z_v(F.relu(F.dropout(self.fc_h_v(x), p=0.2)))  # Value stream
        a = self.fc_z_a(F.relu(F.dropout(self.fc_h_a(x), p=0.2)))  # Advantage stream
        # v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        # a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=-1)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=-1)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name and name != 'fc':
                module.reset_noise()
