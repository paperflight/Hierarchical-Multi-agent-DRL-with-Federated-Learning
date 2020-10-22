# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
import global_parameters as gp


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, indim, block, layers):
        super(ResNet, self).__init__()
        self.conv = conv3x3(indim, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = 16
        self.conv1 = self.make_layer(block, 16, layers[0])
        self.conv2 = self.make_layer(block, 32, layers[0], 2)
        self.conv3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(2)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avg_pool(out)
        return out


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
        self.atoms = args.atoms
        self.action_space = action_space
        self.archit = args.architecture

        if 'canonical' in args.architecture and '61obv' in args.architecture and '2uav' in args.architecture:
            self.convs = nn.Sequential(nn.Conv2d(args.history_length_accesspoint, 16, 8, stride=3, padding=2), nn.LeakyReLU(),
                                       nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Dropout2d(0.2))
            self.conv_output_size = 2368  # 41: 2: 1600  # 61: 2: 2368 3: 3200 4: 4288  # 4 uav: 4992
        elif 'canonical' in args.architecture and '41obv' in args.architecture and '2uav' in args.architecture:
            self.convs = nn.Sequential(nn.Conv2d(args.history_length_accesspoint, 16, 8, stride=3, padding=2), nn.LeakyReLU(),
                                       nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Dropout2d(0.2))
            self.conv_output_size = 1600  # 41: 2: 1600
        elif 'canonical' in args.architecture and '61obv' in args.architecture and '4uav' in args.architecture:
            self.convs = nn.Sequential(nn.Conv2d(args.history_length_accesspoint, 16, 8, stride=3, padding=2), nn.LeakyReLU(),
                                       nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       # nn.MaxPool2d((args.dense_of_uav, 1)),
                                       nn.Dropout2d(0.2))
            self.conv_output_size = 3648
            # 41: 2: 1600  # 61: 2: 2368 3: 3200 4: 4288  # 4 uav: 4992 /pooling 1216/ 3dim obs 3648
        elif args.architecture == 'canonical_3d':
            self.convs = nn.Sequential(nn.Conv3d(1, 32, (gp.OBSERVATION_DIMS, 8, 8), stride=(gp.OBSERVATION_DIMS, 3, 3),
                                                 padding=(0, 2, 2)), nn.LeakyReLU(),
                                       nn.Conv3d(32, 64, (gp.NUM_OF_UAV, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)), nn.LeakyReLU(),
                                       nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=0), nn.LeakyReLU())
            self.conv_output_size = 12160  # 2: 12160 3: 3200 4: 4288
        elif args.architecture == 'resnet8':
            net_args = {
                "indim": gp.OBSERVATION_DIMS * gp.NUM_OF_UAV,
                "block": ResidualBlock,
                "layers": [2, 2, 2, 2]
            }
            self.convs = ResNet(**net_args)
            self.conv_output_size = 64 * 4 * 4
        elif 'data-efficient' in args.architecture and '61obv' in args.architecture and '4uav' in args.architecture:
            self.convs = nn.Sequential(nn.Conv2d(args.history_length_accesspoint, 16, 5, stride=3, padding=2, dilation= 2), nn.LeakyReLU(),
                                       nn.Conv2d(16, 32, 3, stride=1, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(), nn.MaxPool2d(2),
                                       nn.Conv2d(32, 32, 3, stride=1, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.MaxPool2d((2, 1)),
                                       nn.Dropout2d(0.2))
            self.conv_output_size = 1248  # 2: 12160 3: 3200 4: 4288
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
        self.fc = nn.Sequential(nn.Linear(self.conv_output_size, args.hidden_size), nn.Dropout(0.2), nn.LeakyReLU())
        self.fc_h_v = NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        # if x.shape[1] != 1:
        #     list_x = torch.split(x, 1, dim=1)
        #     x = torch.cat(list_x, dim=3)
        if '3d' in self.archit:
            x = x.unsqueeze(1)
        if 'resnet' in self.archit:
            x = x.squeeze(1)
        x = self.convs(x.float())
        x = self.fc(x.view(x.size(0), -1))
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
