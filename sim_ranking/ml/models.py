from typing import Dict, Sequence, List

import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import numpy as np
import einops


class PairWiseModel(nn.Module):
    def __init__(
        self,
        fc_units: List[int],
        n_scalar_inputs: int,
        n_ims: int,
    ):
        super().__init__()

        self.n_fc_layers = len(fc_units)

        fc_input_size = (5 * n_ims) + n_scalar_inputs
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_units)):
            if i == 0:
                self.fc_layers.append(nn.Linear(fc_input_size, fc_units[i]))
            else:
                self.fc_layers.append(nn.Linear(fc_units[i - 1], fc_units[i]))

            self.fc_layers.append(nn.ELU())

        self.fc_layers.append(nn.Linear(self.fc_layers[-2].out_features, 1))
        # self.fc_layers.append(nn.Sigmoid())

    def forward(self, im_values: torch.Tensor, scalar_values: torch.Tensor):
        # Flatten
        im_values = torch.reshape(im_values, (im_values.shape[0], -1))
        x = torch.cat((im_values, scalar_values), 1)

        return self.fc_layers(x)

class ProbIndModel(nn.Module):

    def __init__(self,
                 fc_units: Sequence[int],
                 n_scalar_inputs: int,
                 n_ims: int,
                 ):
        super().__init__()

        self.fc_units = fc_units
        self.n_scalar_inputs = n_scalar_inputs
        self.n_ims = n_ims

        input_size = n_ims * 3 + n_scalar_inputs

        self.fc_layers = nn.Sequential()
        for i in range(len(fc_units)):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_size, fc_units[i]))
            else:
                self.fc_layers.append(nn.Linear(fc_units[i - 1], fc_units[i]))

            if i == len(fc_units) - 1:
                self.fc_layers.append(nn.Linear(fc_units[i], 1))
            else:
                self.fc_layers.append(nn.BatchNorm1d(self.fc_units[i]))
                self.fc_layers.append(nn.LeakyReLU())

    def forward(self, im_values: torch.Tensor, scalar_values: torch.Tensor):
        X_im = einops.rearrange(im_values, "batch type rel im -> batch rel (type im)")
        X_ss = einops.repeat(scalar_values, "batch ss -> batch rel ss", rel=im_values.shape[2])

        X = torch.cat((X_im, X_ss), axis=2)
        X = einops.rearrange(X, "batch rel feature -> (batch rel) feature")

        X = self.fc_layers(X)
        X = einops.rearrange(X, "(batch rel) 1 -> batch rel", batch=im_values.shape[0], rel=im_values.shape[2])

        X = custom_sigmoid(X, 0.5)

        pred = X / X.sum(axis=1, keepdims=True)
        return pred


class ProbCombModel(nn.Module):

    def __init__(self,
                 fc_im_units: Sequence[int],
                 n_ims: int,
                 fc_ss_units: Sequence[int],
                 fc_comb_units: Sequence[int],
                 n_ss_inputs: int,
                 n_rels: int,
                 ):
        super().__init__()

        self.n_rels = n_rels

        self.fc_im_units = fc_im_units
        self.fc_ss_units = fc_ss_units
        self.fc_comb_units = fc_comb_units

        self.im_input_size = n_ims * 3
        self.n_ss_inputs = n_ss_inputs
        self.comb_input_size = self.fc_im_units[-1] * self.n_rels + self.fc_ss_units[-1]



        # IM layers
        self.fc_im_layers = nn.Sequential()
        for i in range(len(self.fc_im_units)):
            if i == 0:
                self.fc_im_layers.append(nn.Linear(self.im_input_size, self.fc_im_units[i]))
            else:
                self.fc_im_layers.append(nn.Linear(self.fc_im_units[i - 1], self.fc_im_units[i]))

            self.fc_im_layers.append(nn.ELU())

        # Site-to-site layers
        self.fc_ss_layers = nn.Sequential()
        for i in range(len(self.fc_ss_units)):
            if i == 0:
                self.fc_ss_layers.append(nn.Linear(self.n_ss_inputs, self.fc_ss_units[i]))
            else:
                self.fc_ss_layers.append(nn.Linear(self.fc_ss_units[i - 1], self.fc_ss_units[i]))

            self.fc_ss_layers.append(nn.ELU())

        # Combined layers
        self.fc_comb_layers = nn.Sequential()
        for i in range(len(self.fc_comb_units)):
            if i == 0:
                self.fc_comb_layers.append(nn.Linear(self.comb_input_size, self.fc_comb_units[i]))
            else:
                self.fc_comb_layers.append(nn.Linear(self.fc_comb_units[i - 1], self.fc_comb_units[i]))

            self.fc_comb_layers.append(nn.ELU())

        self.fc_comb_layers.append(nn.Linear(self.fc_comb_units[-1], self.n_rels))

    def forward(self, im_values: torch.Tensor, ss_values: torch.Tensor):
        X_im = []
        for i in range(self.n_rels):
            cur_im_values = einops.rearrange(im_values[:, :, i, :], "batch type im -> batch (type im)")
            X_im.append(self.fc_im_layers(cur_im_values))

        X_im = torch.cat(X_im, axis=1)
        X_ss = self.fc_ss_layers(ss_values)

        X_comb = torch.cat((X_im, X_ss), axis=1)
        X_comb = self.fc_comb_layers(X_comb)

        # X_comb = F.sigmoid(X_comb)
        X_comb = custom_sigmoid(X_comb, 1.0)

        # Normalise
        return X_comb / X_comb.sum(axis=1, keepdims=True)


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

        dist_term = torch.exp(-((dist[:, None] / self.dist_scale) ** self.dist_exp))

        angular_term = (1 + angular_dist[:, None] / self.angular_scale) * (
            1 - angular_dist[:, None] / torch.pi
        ) ** (np.pi / self.angular_scale)

        vs30_term = torch.exp(-vs30_dist[:, None] / self.vs30_scale)

        result = dist_term * (
            self.weight * angular_term + (1 - self.weight) * vs30_term
        )
        return result


class MLPModel(nn.Module):
    def __init__(self, n_outputs: int, units: Sequence[int], n_scalar_inputs: int):
        super().__init__()
        self.units = units
        self.layers = nn.Sequential()
        for i in range(len(units)):
            if i == 0:
                self.layers.append(nn.Linear(n_scalar_inputs, units[i]))
            else:
                self.layers.append(nn.Linear(units[i - 1], units[i]))
            self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(units[-1], n_outputs))

    def forward(self, x):
        return self.layers(x)


def custom_sigmoid(x: torch.Tensor, a: float):
    return 1 / (1 + torch.exp(-a * x))