# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


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


class Conv_LSTM_Cell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, stride, padding, dilation):
        super(Conv_LSTM_Cell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if type(stride) == int:
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if type(dilation) == int:
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
        if type(padding) == int:
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.number_features = 4

        self.wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding,
                             self.dilation, bias=True)
        self.whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, bias=False)
        self.wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding,
                             self.dilation, bias=True)
        self.whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, bias=False)
        self.wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding,
                             self.dilation, bias=True)
        self.whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, bias=False)
        self.wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding,
                             self.dilation, bias=True)
        self.who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, bias=False)

        self.wci = None
        self.wcf = None
        self.wco = None

    def forward(self, x, h, c):
        # a = self.wxi(x)
        # b = self.whi(h)
        # e = c * self.wci
        ci = torch.sigmoid(self.wxi(x) + self.whi(h) + c * self.wci)
        cf = torch.sigmoid(self.wxf(x) + self.whf(h) + c * self.wcf)
        cc = cf * c + ci * torch.tanh(self.wxc(x) + self.whc(h))
        co = torch.sigmoid(self.wxo(x) + self.who(h) + cc * self.wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        shape_after = (math.floor((shape[0] + self.padding[0] * 2 - self.dilation[0] *
                                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1),
                       math.floor((shape[1] + self.padding[0] * 2 - self.dilation[1] *
                                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))

        if self.wci is None:
            self.wci = Variable(torch.zeros(1, hidden, shape_after[0], shape_after[1]))
            self.wcf = Variable(torch.zeros(1, hidden, shape_after[0], shape_after[1]))
            self.wco = Variable(torch.zeros(1, hidden, shape_after[0], shape_after[1]))
        else:
            assert shape_after[0] == self.wci.size()[2], 'Input Height Mismatched!'
            assert shape_after[1] == self.wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape_after[0], shape_after[1])),
                Variable(torch.zeros(batch_size, hidden, shape_after[0], shape_after[1])))


class Conv_LSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step, stride, padding, dilation, effective_step=[1]):
        super(Conv_LSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_layers = len(hidden_channels)
        self.step = step
        self.dilation = dilation
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = Conv_LSTM_Cell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i], self.stride[i],
                                  self.padding[i], self.dilation[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input_obs):
        internal_state = []
        outputs = []
        x, new_c = None, None
        split_input = torch.split(input_obs, 1, dim=1)
        for step, x in enumerate(split_input):
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space

        if 'canonical' in args.architecture and '61obv' in args.architecture:
            self.convs = nn.Sequential(Conv_LSTM(input_channels=1, hidden_channels=[16, 32, 64, 64],
                                                 kernel_size=[8, 4, 3, 3], step=4, stride=[3, 2, 1, 1], padding=[2, 1, 0, 0],
                                                 dilation=[1, 1, 1, 1], effective_step=[1]))
            self.conv_output_size = 2368  # 2: 2048  4: 4288  # 61 obs: 2: 2368  # 4 uav: 4992
            # TODO: Reduce one layer if use 41 obs
        elif 'canonical' in args.architecture and '41obv' in args.architecture:
            self.convs = nn.Sequential(Conv_LSTM(input_channels=1, hidden_channels=[16, 32, 64, 64],
                                                 kernel_size=[8, 4, 3, 3], step=4, stride=[3, 2, 1, 1], padding=[2, 1, 1, 0],
                                                 dilation=[1, 1, 1, 1], effective_step=[1]))
            self.conv_output_size = 1600
            # TODO: Reduce one layer if use 41 obs
        elif 'canonical' in args.architecture and '61obv' in args.architecture and '4uav' in args.architecture:
            self.convs = nn.Sequential(Conv_LSTM(input_channels=1, hidden_channels=[16, 32, 64, 64],
                                                 kernel_size=[8, 4, 3, 3], step=4, stride=[3, 2, 1, 1], padding=[2, 1, 0, 0],
                                                 dilation=[1, 1, 1, 1], effective_step=[1]))
            self.conv_output_size = 4992  # 41: 2: 1600  # 61: 2: 2368 3: 3200 4: 4288  # 4 uav: 4992
        # if args.architecture == 'canonical':
        #     self.convs = nn.Sequential(Conv_LSTM(input_channels=1, hidden_channels=[32, 64, 64],
        #                                          kernel_size=[8, 4, 3], step=4, stride=[3, 2, 1], padding=[2, 1, 0],
        #                                          effective_step=[3]))
        #     self.conv_output_size = 3200
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
        self.fc = nn.Sequential(nn.Linear(self.conv_output_size, args.hidden_size), nn.Dropout(0.2), nn.LeakyReLU())
        self.fc_h_v = NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        x, _ = self.convs(x.float())
        x = self.fc(x[0].view(x[0].size(0), -1))
        v = self.fc_z_v(F.relu(F.dropout(self.fc_h_v(x), p=0.2)))  # Value stream
        a = self.fc_z_a(F.relu(F.dropout(self.fc_h_a(x), p=0.2)))  # Advantage stream
        # v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        # a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name and name != 'fc':
                module.reset_noise()
