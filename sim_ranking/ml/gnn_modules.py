from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gdata
import torch_geometric.nn as gnn
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.nn.inits as ginits
import numpy as np


class CustomGNN(torch.nn.Module):

    def __init__(
        self,
        n_obs_node_features: int,
        n_int_node_features: int,
        n_edge_features: int,
        n_int_node_channels: int,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            gnn.HeteroConv(
                {
                    ("site_obs", "informs", "site_int"): CustomConv(
                        in_channels=(n_obs_node_features, n_int_node_features),
                        out_channels=n_int_node_channels,
                        nn_model=nn.Sequential(
                            nn.Linear(n_edge_features, 8),
                            nn.ReLU(),
                            nn.Linear(8, n_obs_node_features * n_int_node_channels),
                        ),
                        bias=True,
                        aggr="add",
                    )
                },
                aggr="sum",
            )
        )

        self.fc1 = nn.Linear(n_int_node_channels, 8)
        self.out_fc = nn.Linear(8, 1)

    def forward(self, data: gdata.Data):
        for cur_conv in self.convs:
            x_dict = cur_conv(
                data.x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict
            )
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_site_int = x_dict["site_int"]

        x = F.relu(self.fc1(x_site_int))
        out = self.out_fc(x)

        return out




class CustomConv(MessagePassing):

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

    # noinspection PyMethodOverriding
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
