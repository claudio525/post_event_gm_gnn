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

        self.fc_layers.append(nn.Linear(self.fc_layers[-2].out_features, 1))
        self.fc_layers.append(nn.Sigmoid())

    def forward(self, rs_int_sim, rs_obs_sim, rs_obs_obs, scalar_features):
        rs_int_sim_out = self.rs_layers(rs_int_sim)
        rs_obs_sim_out = self.rs_layers(rs_obs_sim)
        rs_obs_obs_out = self.rs_layers(rs_obs_obs)

        rs_conv_out = torch.cat(
            (rs_int_sim_out, rs_obs_sim_out, rs_obs_obs_out, scalar_features), 1
        )

        return self.fc_layers(rs_conv_out)

