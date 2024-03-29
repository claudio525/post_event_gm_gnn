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


class ProbModel(nn.Module):
    """Base class, don't use directly"""

    def __init__(
        self,
        n_scalar_inputs: int,
        n_ims: int,
        n_im_features: int,
    ):

        self.n_scalar_inputs = n_scalar_inputs
        self.n_ims = n_ims
        self.n_im_features = n_im_features

        super().__init__()


class ProbIndModel(ProbModel):
    def __init__(
        self,
        fc_units: Sequence[int],
        n_scalar_inputs: int,
        n_ims: int,
        n_im_features: int,
        is_sub_model: bool,
        per_im_prob: bool
    ):
        super().__init__(
            n_scalar_inputs=n_scalar_inputs,
            n_ims=n_ims,
            n_im_features=n_im_features,
        )

        self.fc_units = fc_units
        self.is_sub_model = is_sub_model
        self.per_im_prob = per_im_prob
        self.input_size = n_ims * self.n_im_features + self.n_scalar_inputs

        self.fc_layers = nn.Sequential()
        for i in range(len(fc_units)):
            if i == 0:
                self.fc_layers.append(nn.Linear(self.input_size, fc_units[i]))
            else:
                self.fc_layers.append(nn.Linear(fc_units[i - 1], fc_units[i]))

            self.fc_layers.append(nn.BatchNorm1d(self.fc_units[i]))
            self.fc_layers.append(nn.LeakyReLU())

        if self.is_sub_model:
            # If the model is a sub-model the last number in fc_units
            # is the number of outputs
            pass
        elif per_im_prob:
            self.fc_layers.append(nn.Linear(fc_units[-1], n_ims))
        else:
            self.fc_layers.append(nn.Linear(fc_units[-1], 1))

    @property
    def n_outputs(self):
        if self.is_sub_model:
            return self.fc_units[-1]
        elif self.per_im_prob:
            return self.n_ims
        return 1

    def forward(self, im_values: torch.Tensor, scalar_values: torch.Tensor):
        X_im = einops.rearrange(im_values, "batch type rel im -> batch rel (type im)")
        X_ss = scalar_values
        # X_ss = einops.repeat(
        #     scalar_values, "batch ss -> batch rel ss", rel=im_values.shape[2]
        # )

        X = torch.cat((X_im, X_ss), axis=2)
        X = einops.rearrange(X, "batch rel feature -> (batch rel) feature")

        X = self.fc_layers(X)
        X = einops.rearrange(
            X,
            "(batch rel) n_outs -> batch rel n_outs",
            batch=im_values.shape[0],
            rel=im_values.shape[2],
            n_outs=self.n_outputs,
        )

        # Apply sigmoid unless its a sub-model
        if not self.is_sub_model:
            X = custom_sigmoid(X.squeeze(), 0.5)
            pred = X / X.sum(axis=1, keepdims=True)
        else:
            pred = X

        return pred


class ProbIMModel(nn.Module):

    def __init__(self, n_inputs: int, units: Sequence[int], n_rels: int):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_rels = n_rels

        self.fc_layers = nn.Sequential()
        for i in range(len(units)):
            if i == 0:
                self.fc_layers.append(nn.Linear(self.n_inputs, units[i]))
            else:
                self.fc_layers.append(nn.Linear(units[i - 1], units[i]))

            self.fc_layers.append(nn.BatchNorm1d(units[i]))
            self.fc_layers.append(nn.LeakyReLU())

        self.fc_layers.append(nn.Linear(units[-1], self.n_rels))
        self.fc_layers.append(nn.Softmax(dim=1))

    def forward(self, im_features: torch.Tensor, scalar_features: torch.Tensor):
        n_ims = im_features.shape[3]

        # Convert shapes that the model performs one prediction per IM
        ## TODO: Add one-hot encoding
        X_im = einops.rearrange(im_features, "batch imf rel im -> (batch im) (rel imf)")
        X_ss = einops.repeat(scalar_features[:, 0, :], "batch ss -> (batch im) ss",
                             im=n_ims)
        X = torch.cat([X_im, X_ss], dim=1)

        pred = self.fc_layers(X)

        # Convert result to shape (batch, n_rels, n_ims)
        pred = einops.rearrange(pred, "(batch im) rel -> batch rel im", im=n_ims)
        return pred




class ProbCombModel(ProbModel):

    def __init__(
        self,
        ind_fc_units: Sequence[int],
        comb_fc_units: Sequence[int],
        n_scalar_inputs: int,
        n_ims: int,
        n_im_features: int,
        per_im_prob: bool
    ):
        super().__init__(
            n_scalar_inputs=n_scalar_inputs,
            n_ims=n_ims,
            n_im_features=n_im_features,
        )

        self.ind_model = ProbIndModel(
            fc_units=ind_fc_units,
            n_scalar_inputs=n_scalar_inputs,
            n_ims=n_ims,
            n_im_features=n_im_features,
            is_sub_model=True,
            per_im_prob=False
        )

        self.fc_comb_units = comb_fc_units

        self.n_inputs = self.ind_model.n_outputs
        self.n_outputs = n_ims if per_im_prob else 1

        # Combined layers
        # self.fc_comb_layers = nn.Sequential()
        self.fc_comb_layers = nn.ModuleList()
        for i in range(len(self.fc_comb_units)):
            if i == 0:
                self.fc_comb_layers.append(
                    nn.Linear(self.n_inputs, self.fc_comb_units[i])
                )
            else:
                self.fc_comb_layers.append(
                    nn.Linear(self.fc_comb_units[i - 1], self.fc_comb_units[i])
                )

            # if i < len(self.fc_comb_units) - 1:
            self.fc_comb_layers.append(nn.BatchNorm1d(self.fc_comb_units[i]))
            self.fc_comb_layers.append(nn.LeakyReLU())

            # self.fc_comb_layers.append(nn.ELU())

        # self.fc_comb_layers.append(nn.Linear(self.fc_comb_units[-1], self.n_rels))
        self.fc_comb_layers.append(nn.Linear(self.fc_comb_units[-1], self.n_outputs))

    def forward(self, im_values: torch.Tensor, scalar_values: torch.Tensor):
        X = self.ind_model(im_values, scalar_values)

        X = einops.rearrange(X, "batch rel feature -> (batch rel) feature")
        for cur_layer in self.fc_comb_layers:
            X = cur_layer(X)
        X = einops.rearrange(X, "(batch rel) n_outs -> batch rel n_outs", batch=im_values.shape[0], rel=im_values.shape[2])

        X = custom_sigmoid(X.squeeze(), 0.5)
        pred = X / X.sum(axis=1, keepdims=True)

        #
        # if len(X.shape) == 3:
        #     X = einops.rearrange(X, "batch rel n_outs -> batch (rel n_outs)")
        #
        # X = self.fc_comb_layers(X)
        #
        # # X = custom_sigmoid(X.squeeze(), 0.5)

        return pred


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

class WeightModel(nn.Module):
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
        X = self.layers(x)

        pred = F.sigmoid(X)
        return pred


def custom_sigmoid(x: torch.Tensor, a: float):
    return 1 / (1 + torch.exp(-a * x))
