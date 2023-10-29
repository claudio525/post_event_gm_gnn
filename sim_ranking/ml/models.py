from typing import Dict, Sequence, List

import torch
from torch import nn
import pandas as pd
import numpy as np


class ResponseSpectrumSimModel(nn.Module):
    def __init__(
        self,
        rs_kernel_sizes: List[int],
        rs_n_channels: List[int],
        rs_padding: List[int],
        fc_units: List[int],
        rs_input_length: int,
        n_scalar_inputs: int,
        n_outputs: int,
        apply_sigmoid: bool = True,
    ):
        super().__init__()

        self.n_rs_layers = len(rs_kernel_sizes)
        self.n_fc_layers = len(fc_units)

        ### Add the convolutional layers
        self.rs_layers = nn.Sequential()
        for i in range(self.n_rs_layers):
            self.rs_layers.append(
                nn.Conv1d(
                    rs_n_channels[i],
                    rs_n_channels[i + 1],
                    rs_kernel_sizes[i],
                    padding=rs_padding[i],
                )
            )
            self.rs_layers.append(nn.ELU())
            self.rs_layers.append(nn.MaxPool1d(2))
        self.rs_layers.append(nn.Flatten())

        ### Add the fully connected layers
        # Get the conv out size
        conv_out_size = self.rs_layers(torch.zeros(1, 1, rs_input_length)).shape[-1]
        fc_input_size = (conv_out_size * 3) + n_scalar_inputs
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_units)):
            if i == 0:
                self.fc_layers.append(nn.Linear(fc_input_size, fc_units[i]))
            else:
                self.fc_layers.append(nn.Linear(fc_units[i - 1], fc_units[i]))

            self.fc_layers.append(nn.ELU())

        self.fc_layers.append(nn.Linear(self.fc_layers[-2].out_features, n_outputs))
        if apply_sigmoid:
            self.fc_layers.append(nn.Sigmoid())

    def forward(self, rs_int_sim, rs_obs_sim, rs_obs_obs, scalar_features):
        rs_int_sim_out = self.rs_layers(rs_int_sim)
        rs_obs_sim_out = self.rs_layers(rs_obs_sim)
        rs_obs_obs_out = self.rs_layers(rs_obs_obs)

        rs_conv_out = torch.cat(
            (rs_int_sim_out, rs_obs_sim_out, rs_obs_obs_out, scalar_features), 1
        )

        return self.fc_layers(rs_conv_out)


class SimpleWeightModel(nn.Module):
    def __init__(self, n_periods: int):
        super().__init__()
        self.linear = nn.Linear(1, n_periods)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class ExpWeightModel(nn.Module):
    """
    The functional form of this model is based on the
    2023 Bodemann et al. spatial correlation model
    """

    def __init__(self, n_periods: int):
        super().__init__()

        self.dist_scale = nn.Parameter(torch.ones(n_periods) * 20)
        self.dist_exp = nn.Parameter(torch.ones(n_periods) * 1)

        self.angular_scale = nn.Parameter(torch.ones(n_periods) * 1.0)

        self.vs30_scale = nn.Parameter(torch.ones(n_periods) * 200.0)

        self.weight = nn.Parameter(torch.ones(n_periods) * 0.5)

    def forward(self, x: torch.Tensor):
        """Expects inputs to have shape [batch_size, 3] where
        the columns are distance, vs30, and angular distance"""
        dist = x[:, 0]
        vs30_dist = x[:, 1]
        angular_dist = x[:, 2]

        dist_term = torch.exp(-(dist[:, None] / self.dist_scale) ** self.dist_exp)

        angular_term = (1 + angular_dist[:, None] / self.angular_scale) * (
            1 - angular_dist[:, None] / torch.pi
        ) ** (np.pi / self.angular_scale)

        vs30_term = torch.exp(-vs30_dist[:, None] / self.vs30_scale)

        result = dist_term * (self.weight * angular_term + (1 - self.weight) * vs30_term)
        return result


class MLPWeightModel(nn.Module):
    def __init__(self, n_periods: int, units: Sequence[int], n_scalar_inputs: int):
        super().__init__()
        self.units = units
        self.layers = nn.Sequential()
        for i in range(len(units)):
            if i == 0:
                self.layers.append(nn.Linear(n_scalar_inputs, units[i]))
            else:
                self.layers.append(nn.Linear(units[i - 1], units[i]))
            self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(units[-1], n_periods))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.layers(x))
