from typing import Any, Sequence, TYPE_CHECKING

import einops
import pandas as pd
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

import ml_tools as mlt

if TYPE_CHECKING:
    from . import gnn_gm


def _create_single_mlp(
    n_inputs: int, n_outputs: int, act_fn_str: str | None, bias: bool = True
) -> nn.Sequential:
    mlp = nn.Sequential(
        nn.Linear(n_inputs, n_outputs, bias=bias),
    )
    if act_fn_str is not None:
        mlp.append(mlt.torch.get_act_fn_layer(act_fn_str))

    return mlp


def _create_multi_mlp(
    n_inputs: int,
    n_units: Sequence[int],
    n_outputs: int,
    act_fn_str: str | None,
    bias: bool = True,
) -> nn.Sequential:
    mlp = nn.Sequential()
    for ix, cur_n_units in enumerate(n_units):
        mlp.append(
            nn.Linear(
                n_inputs if ix == 0 else n_units[ix - 1],
                cur_n_units,
                bias=bias,
            )
        ),
        if act_fn_str is not None:
            mlp.append(mlt.torch.get_act_fn_layer(act_fn_str))
    mlp.append(
        nn.Linear(
            n_units[-1],
            n_outputs,
            bias=bias,
        )
    )

    return mlp


class CustomAttentionGNN(torch.nn.Module):

    def __init__(
        self,
        # n_obs_node_features: int,
        # n_int_node_features: int,
        # n_edge_features: int,
        run_config: "gnn_gm.RunConfig",
    ):
        super().__init__()
        self.run_config = run_config

        # Sanity check
        assert len(run_config.n_int_node_channels) == len(
            run_config.n_obs_node_channels
        ) and len(run_config.n_int_node_channels) == len(run_config.n_att_heads)
        self.n_convs = len(self.run_config.n_int_node_channels)

        self.convs = torch.nn.ModuleList()
        for ix, (cur_n_att_heads, cur_int_n_channels, cur_obs_n_channels) in enumerate(
            zip(
                run_config.n_att_heads,
                run_config.n_int_node_channels,
                run_config.n_obs_node_channels,
            )
        ):
            ## Observation node update model
            # This model updates the observation nodes via a self loop
            n_obs_update_in_channels = (
                run_config.site_obs_n_features
                if ix == 0
                else run_config.n_obs_node_channels[ix - 1]
            )
            obs_update_model = _create_single_mlp(
                n_obs_update_in_channels,
                cur_obs_n_channels,
                # Use tanh for first layer to handle large nan replacement values
                "tanh" if ix == 0 else run_config.obs_embedding_act_fn,
            )

            ## Observation node transform model
            # This model performs the transformation of the observation
            # nodes used to update the SoI node
            n_obs_transform_in_channels = (
                run_config.site_obs_n_features
                if ix == 0
                else run_config.n_obs_node_channels[ix - 1]
            )
            obs_transform_models = nn.ModuleList(
                [
                    _create_single_mlp(
                        n_obs_transform_in_channels,
                        cur_int_n_channels,
                        # Use tanh for first layer to handle large nan replacement values
                        "tanh" if ix == 0 else run_config.int_embedding_act_fn,
                    )
                    for _ in range(cur_n_att_heads)
                ]
            )

            # SoI node transform models
            n_int_in_channels = (
                run_config.site_int_n_features
                if ix == 0
                else run_config.n_int_node_channels[ix - 1] * run_config.n_att_heads[ix]
            )
            int_transform_model = _create_single_mlp(
                        n_int_in_channels,
                        cur_int_n_channels * run_config.n_att_heads[ix],
                        run_config.int_embedding_act_fn,
                    )

            # Attention model
            att_model = _create_multi_mlp(
                run_config.n_edge_features,
                run_config.att_n_units,
                cur_n_att_heads,
                run_config.att_act_fn,
            )

            self.convs.append(
                gnn.HeteroConv(
                    {
                        ("site_obs", "self_loop", "site_obs"): ObsNodeConv(
                            obs_update_model=obs_update_model,
                        ),
                        ("site_obs", "informs", "site_int"): IntNodeConv(
                            obs_transform_models=obs_transform_models,
                            int_transform_model=int_transform_model,
                            att_model=att_model,
                        ),
                    },
                    aggr="sum",
                )
            )

        # self.fc1 = nn.Linear(run_config.n_int_node_channels[-1], run_config.fc_n_units)
        # self.out_fc = nn.Linear(run_config.fc_n_units, run_config.n_outputs)

        self.out_fc = nn.Linear(
            run_config.n_int_node_channels[-1] * run_config.n_att_heads[-1], run_config.n_outputs
        )

    def get_attention_coeff(self, data: gdata.HeteroData):
        """
        Gets the attention coefficient per layer

        Note: Not part of the prediction or training process,
        only for extraction of attention coefficients!
        """
        rel_data = data[("site_obs", "informs", "site_int")]
        dest_ind = rel_data["edge_index"][1].numpy(force=True)
        sc_id = data["metadata"]["sc_id"]

        attn_coeffs = {sc_id[dest_ix]: [] for dest_ix in np.unique(dest_ind)}
        obs_sites = {
            sc_id[dest_ix]: data["metadata"]["obs_sites"][dest_ix]
            for dest_ix in np.unique(dest_ind)
        }
        for ix, cur_conv in enumerate(self.convs):
            cur_attn_coeffs = (
                cur_conv.convs[("site_obs", "informs", "site_int")]
                .compute_attn_coeffs(rel_data["edge_attr"])
                .numpy(force=True)
            )

            for dest_ix in np.unique(dest_ind):
                attn_coeffs[sc_id[dest_ix]].append(cur_attn_coeffs[dest_ind == dest_ix])

        result = []
        for ix, (key, value) in enumerate(attn_coeffs.items()):
            cur_df = pd.DataFrame(
                index=mlt.array_utils.numpy_str_join("_", key, obs_sites[key]),
                data=einops.rearrange(np.stack(value, axis=2), "obs att conv -> obs (conv att)"),
                columns=[f"conv_{i}_head_{j}" for i in range(self.n_convs) for j in range(self.run_config.n_att_heads[i])],
            )
            cur_df["event"] = data["metadata"]["event"][ix]
            cur_df["obs_site"] = obs_sites[key]
            cur_df["site_int"] = data["metadata"]["site_int"][ix]

            result.append(cur_df)

        # result = {
        #     key: pd.DataFrame(index=mlt.array_utils.numpy_str_join("_", key, obs_sites[key]), data=np.concatenate(value, axis=1))
        #     for key, value in attn_coeffs.items()
        # }
        return pd.concat(result, axis=0)

    def forward(self, data: gdata.HeteroData):
        for ix, cur_conv in enumerate(self.convs):
            # Overwrite with result from previous layer if not the first layer
            x_dict = data.x_dict if ix == 0 else data.x_dict | x_dict

            # Apply convolution
            x_dict = cur_conv(
                x_dict,
                data.edge_index_dict,
                edge_attr_dict=data.edge_attr_dict,
            )

            # Save results
            x_dict = {
                # key: F.dropout(mlt.torch.get_act_fn(self.run_config.gcn_act_fn)(x), p=0.5, training=self.training)
                key: mlt.torch.get_act_fn(self.run_config.gcn_act_fn)(x)
                for key, x in x_dict.items()
            }

        x_site_int = x_dict["site_int"]

        # x = mlt.torch.get_act_fn(self.run_config.fcc_act_fn)(self.fc1(x_site_int))
        # out = self.out_fc(x)

        out = self.out_fc(x_site_int)
        ln_im_mean, ln_im_std = out.chunk(2, dim=1)

        # Clip predicted values to prevent numerical issues
        # ln_im_std = torch.clamp(ln_im_std, min=-20, max=5)
        # ln_im_mean = torch.clamp(ln_im_mean, min=-20, max=5)

        return ln_im_mean, ln_im_std


class ObsNodeConv(MessagePassing):

    def __init__(self, obs_update_model: nn.Module, **kwargs):
        super().__init__(aggr="add", **kwargs)

        self.obs_update_model = obs_update_model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        out = self.obs_update_model(x)
        return out


class IntNodeConv(MessagePassing):

    def __init__(
        self,
        # obs_transform_model: nn.Module,
        # int_transform_model: nn.Module,
        # att_model: nn.Module,
        obs_transform_models: nn.ModuleList,
        int_transform_model: nn.Module,
        att_model: nn.Module,
        **kwargs,
    ):
        """
        Parameters
        ----------
        obs_transform_model: torch.nn.Module
            Transformation model for the source nodes
        int_transform_model: torch.nn.Module
            Transformation model for the target nodes
        att_model: torch.nn.Module
            Self-attention model
        kwargs
        """
        super().__init__(aggr="add", **kwargs)

        self.att_model = att_model

        self.obs_transform_models = obs_transform_models
        self.int_transform_model = int_transform_model

        self.n_heads = len(self.obs_transform_models)

        # self.obs_transform_model = obs_transform_model
        # self.int_transform_model = int_transform_model

        # self.aggr = "add"
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        ginits.reset(self.att_model)
        ginits.reset(self.obs_transform_models)
        ginits.reset(self.int_transform_model)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the messages
        m_s = self.propagate(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
        )

        # Update the nodes
        out = m_s + self.int_transform_model(x[1])
        return out

    def compute_attn_coeffs(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute the attention coefficients"""
        a = self.att_model(edge_attr)

        alpha = torch.sigmoid(a)
        return alpha

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tensor:
        # Compute the attention coefficients
        alpha = self.compute_attn_coeffs(edge_attr)

        # Compute the message
        # 1) Compute the transformed source node
        # 2) Multiply with the attention coefficient
        # 3) Concatenate the results
        m = torch.cat(
            [
                alpha[:, i][:, None] * self.obs_transform_models[i](x_j)
                for i in range(self.n_heads)
            ],
            dim=1,
        )
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

        # x = F.relu(self.fc1(x_site_int))
        # out = self.out_fc(x)

        out = self.out_fc(x_site_int)

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
