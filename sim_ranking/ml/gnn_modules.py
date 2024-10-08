from typing import Any, Sequence, TYPE_CHECKING

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gdata
import torch_geometric.nn as gnn
from torch import Tensor
import torch_geometric.nn.inits as ginits
import torch_geometric.utils as gutils
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Size
import numpy as np

if TYPE_CHECKING:
    from . import gnn_gm


class BasicAttentionGNN(torch.nn.Module):

    def __init__(
        self,
        n_obs_node_features: int,
        n_obs_scalar_node_features: int,
        n_int_node_features: int,
        n_edge_features: int,
        run_config: "gnn_gm.RunConfig",
        site_obs_scalar_feature_ind: torch.Tensor,
    ):
        super().__init__()
        self.run_config = run_config

        self.convs = torch.nn.ModuleList()
        for cur_n_channels in run_config.n_int_node_channels:
            self.convs.append(
                gnn.HeteroConv(
                    {
                        ("site_obs", "informs", "site_int"): BasicAttentionConv(
                            in_channels=(n_obs_node_features, n_int_node_features),
                            out_channels=cur_n_channels,
                            att_model=nn.Sequential(
                                nn.Linear(
                                    n_obs_scalar_node_features
                                    + n_edge_features
                                    + n_int_node_features,
                                    run_config.n_ims,
                                ),
                                nn.LeakyReLU(negative_slope=0.2),
                            ),
                            source_scalar_feature_ind=site_obs_scalar_feature_ind,
                            pred_std=run_config.pred_std,
                        )
                    },
                    aggr="sum",
                )
            )

        self.fc1 = nn.Linear(run_config.n_int_node_channels[-1], run_config.fc_n_units)
        self.out_fc = nn.Linear(run_config.fc_n_units, run_config.n_outputs)

    def forward(self, data: gdata.HeteroData):
        for cur_conv in self.convs:
            x_dict = cur_conv(
                data.x_dict,
                data.edge_index_dict,
                edge_attr_dict=data.edge_attr_dict,
            )
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_site_int = x_dict["site_int"]

        x = F.relu(self.fc1(x_site_int))
        out = self.out_fc(x)

        if self.run_config.pred_std:
            ln_im_mean, ln_im_std = out.chunk(2, dim=1)

            # Clip predicted values prevent numerical issues
            ln_im_std = torch.clamp(ln_im_std, min=-15, max=5)
            ln_im_mean = torch.clamp(ln_im_mean, min=-15, max=5)

            return ln_im_mean, ln_im_std
        return out


class BasicAttentionConv(MessagePassing):

    def __init__(
        self,
        in_channels: tuple[int, int],
        out_channels: int,
        att_model: nn.Module,
        source_scalar_feature_ind: torch.Tensor,
        pred_std: bool = True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        in_channels: tuple
            The input channels for the source and target nodes
        out_channels: int
            The output channels
        att_model: torch.nn.Module
            Self-attention model
        kwargs
        """
        super().__init__(**kwargs)
        self.pred_std = pred_std

        self.in_channels_target = in_channels[1]
        self.in_channels_source = in_channels[0]
        self.out_channels = out_channels

        self.att_model = att_model
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        self.obs_transform = nn.Linear(
            self.in_channels_source, self.out_channels, bias=False
        )
        self.source_transform = nn.Linear(
            self.in_channels_target, self.out_channels, bias=False
        )

        self.source_scalar_feature_ind = source_scalar_feature_ind

        self.aggr = "add"

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        ginits.reset(self.att_model)
        ginits.reset(self.obs_transform)
        ginits.reset(self.source_transform)
        ginits.zeros(self.bias)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the messages
        m_i = self.propagate(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
        )

        # Update the nodes
        out = m_i + self.source_transform(x[1]) + self.bias
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tensor:
        source_ind, dest_ind = edge_index

        # Compute the attention coefficients
        a = self.att_model(
            torch.cat((x_j[:, self.source_scalar_feature_ind], edge_attr, x_i), dim=1)
        )
        # Normalise
        alpha = gutils.softmax(a, dest_ind)

        if self.pred_std:
            m = alpha.tile((1, 2)) * self.obs_transform(x_j)
        else:
            m = alpha * self.obs_transform(x_j)

        return m


class BasicGNN(torch.nn.Module):
    """
    Represents a very basic Graph Neural Network
    Updates the site of interest nodes based on the observation nodes.
    IMs are predicted using an MLP on the site of interest node embeddings.
    """

    def __init__(
        self,
        n_obs_node_features: int,
        n_int_node_features: int,
        n_edge_features: int,
        n_int_node_channels: int,
        n_ims: int,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            gnn.HeteroConv(
                {
                    ("site_obs", "informs", "site_int"): BasicConv(
                        in_channels=(n_obs_node_features, n_int_node_features),
                        out_channels=n_int_node_channels,
                        nn_model=nn.Sequential(
                            nn.Linear(n_edge_features, 16),
                            nn.ReLU(),
                            nn.Linear(16, n_obs_node_features * n_int_node_channels),
                        ),
                        bias=True,
                        aggr="add",
                    )
                },
                aggr="sum",
            )
        )

        self.fc1 = nn.Linear(n_int_node_channels, 16)
        self.out_fc = nn.Linear(16, n_ims)

    def forward(self, data: gdata.HeteroData):
        for cur_conv in self.convs:
            x_dict = cur_conv(
                data.x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict
            )
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_site_int = x_dict["site_int"]

        x = F.relu(self.fc1(x_site_int))
        out = self.out_fc(x)

        return out


class BasicConv(MessagePassing):
    """
    Very simple graph convolutional layer.
    Updates the site of interest nodes based on the observation nodes.
    With the weight matrix in the message function coming from an MLP,
    that uses the edge features as input.
    """

    def __init__(
        self,
        in_channels: tuple[int, int],
        out_channels: int,
        nn_model: torch.nn.Module,
        bias: bool = True,
        aggr: str = "add",
        **kwargs,
    ):
        """
        Parameters
        ----------
        in_channels: tuple
            The input channels for the source and target nodes
        out_channels: int
            The output channels
        nn_model: torch.nn.Module
            The neural network to be used in the message function
        bias: bool, optional
            Whether to use bias in the update function
        aggr: str, optional
            Aggregation method to use
        """
        super().__init__(aggr, **kwargs)

        self.in_channel_target = in_channels[1]
        self.in_channel_source = in_channels[0]
        self.out_channels = out_channels
        self.nn_model = nn_model
        self.aggr = aggr

        self.source_lin = nn.Linear(
            self.in_channel_target, self.out_channels, bias=False
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        ginits.reset(self.nn_model)
        self.source_lin.reset_parameters()
        ginits.zeros(self.bias)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x)
        assert out.shape[0] == x[1].shape[0]

        out = out + self.source_lin(x[1])

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.nn_model(edge_attr)
        weights = weights.view(-1, self.in_channel_source, self.out_channels)

        # Batched matrix multiplication
        messages = einops.einsum(x_j, weights, "b i, b i j -> b j")
        return messages
