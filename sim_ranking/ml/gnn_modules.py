import random
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
from torch_geometric.transforms import BaseTransform
import numpy as np

import ml_tools as mlt

if TYPE_CHECKING:
    from . import gnn_gm


def _create_single_mlp(
    n_inputs: int, n_outputs: int, act_fn_str: str | None, bias: bool = True
) -> nn.Sequential:
    """Creates a single-layer perceptron"""
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
    """Creates a multi-layer perceptron"""
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
        run_config: "gnn_gm.RunConfig",
    ):
        super().__init__()
        self.run_config = run_config

        # Sanity check
        assert len(run_config.n_int_node_channels) == len(
            run_config.n_obs_node_channels
        ) and len(run_config.n_int_node_channels) == len(
            run_config.n_att_heads
        ), "Number of layers (SoI and Obs), and attention heads must be equal"
        self.n_convs = len(self.run_config.n_int_node_channels)

        self.convs = torch.nn.ModuleList()
        self.int_bns = torch.nn.ModuleList() if run_config.batch_norm else None
        self.obs_bns = torch.nn.ModuleList() if run_config.batch_norm else None
        self.edge_bns = torch.nn.ModuleList() if run_config.batch_norm else None
        for ix, (
            cur_n_att_heads,
            cur_int_n_channels,
            cur_obs_n_channels,
            cur_n_edge_channels,
        ) in enumerate(
            zip(
                run_config.n_att_heads,
                run_config.n_int_node_channels,
                run_config.n_obs_node_channels,
                run_config.n_edge_channels,
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
            n_int_out_channels = cur_int_n_channels * run_config.n_att_heads[ix]
            int_update_model = _create_single_mlp(
                n_int_in_channels,
                n_int_out_channels,
                run_config.int_embedding_act_fn,
            )

            n_edge_in_channels = (
                run_config.n_edge_features
                if ix == 0
                else run_config.n_edge_channels[ix - 1]
            )
            edge_update_model = _create_single_mlp(
                n_edge_in_channels,
                cur_n_edge_channels,
                run_config.edge_embedding_act_fn,
            )

            # Attention model
            att_model = _create_multi_mlp(
                cur_n_edge_channels,
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
                            int_update_model=int_update_model,
                            edge_update_model=edge_update_model,
                            att_model=att_model,
                        ),
                    },
                    aggr="sum",
                )
            )

            if run_config.batch_norm:
                self.int_bns.append(nn.BatchNorm1d(n_int_out_channels))
                self.obs_bns.append(nn.BatchNorm1d(cur_obs_n_channels))
                self.edge_bns.append(nn.BatchNorm1d(cur_n_edge_channels))

        if run_config.fc_n_units is None:
            self.fc1 = None
            self.out_fc = nn.Linear(
                run_config.n_int_node_channels[-1] * run_config.n_att_heads[-1],
                run_config.n_outputs,
            )
        else:
            self.fc1 = nn.Linear(
                run_config.n_int_node_channels[-1] * run_config.n_att_heads[-1],
                run_config.fc_n_units,
            )
            self.out_fc = nn.Linear(run_config.fc_n_units, run_config.n_outputs)

    @property
    def n_train_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        prev_edge_attrs = rel_data["edge_attr"]
        for ix, cur_conv in enumerate(self.convs):
            cur_conv_module = cur_conv.convs[("site_obs", "informs", "site_int")]
            cur_edge_attrs = cur_conv_module.edge_update(prev_edge_attrs)
            cur_attn_coeffs = cur_conv_module.compute_attn_coeffs(cur_edge_attrs).numpy(
                force=True
            )

            for dest_ix in np.unique(dest_ind):
                attn_coeffs[sc_id[dest_ix]].append(cur_attn_coeffs[dest_ind == dest_ix])

            prev_edge_attrs = cur_edge_attrs

        result = []
        for ix, (key, value) in enumerate(attn_coeffs.items()):
            cur_df = pd.DataFrame(
                index=mlt.array_utils.numpy_str_join("_", key, obs_sites[key]),
                data=einops.rearrange(
                    np.stack(value, axis=2), "obs att conv -> obs (conv att)"
                ),
                columns=[
                    f"conv_{i}_head_{j}"
                    for i in range(self.n_convs)
                    for j in range(self.run_config.n_att_heads[i])
                ],
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
        node_emb_dict = data.x_dict
        edge_emb_dict = data.edge_attr_dict

        for ix, cur_conv in enumerate(self.convs):
            # Apply convolution
            x_dict = cur_conv(
                node_emb_dict,
                data.edge_index_dict,
                edge_attr_dict=edge_emb_dict,
            )

            # Get updated embeddings
            obs_node_embedding = x_dict["site_obs"]
            int_node_embedding = x_dict["site_int"][0]
            obs_int_edge_embedding = x_dict["site_int"][1]

            # Apply activation function
            node_emb_dict["site_obs"] = mlt.torch.get_act_fn(
                self.run_config.gcn_act_fn
            )(obs_node_embedding)
            node_emb_dict["site_int"] = mlt.torch.get_act_fn(
                self.run_config.gcn_act_fn
            )(int_node_embedding)
            edge_emb_dict[("site_obs", "informs", "site_int")] = mlt.torch.get_act_fn(
                self.run_config.gcn_act_fn
            )(obs_int_edge_embedding)

            # Apply batch normalization
            if self.run_config.batch_norm:
                node_emb_dict["site_obs"] = self.obs_bns[ix](node_emb_dict["site_obs"])
                node_emb_dict["site_int"] = self.int_bns[ix](node_emb_dict["site_int"])
                edge_emb_dict[("site_obs", "informs", "site_int")] = self.edge_bns[ix](
                    edge_emb_dict[("site_obs", "informs", "site_int")]
                )

            # Apply dropout
            if self.run_config.dropout_rate > 0:
                node_emb_dict["site_obs"] = F.dropout(
                    node_emb_dict["site_obs"],
                    p=self.run_config.dropout_rate,
                    training=self.training,
                )
                node_emb_dict["site_int"] = F.dropout(
                    node_emb_dict["site_int"],
                    p=self.run_config.dropout_rate,
                    training=self.training,
                )
                edge_emb_dict[("site_obs", "informs", "site_int")] = F.dropout(
                    edge_emb_dict[("site_obs", "informs", "site_int")],
                    p=self.run_config.dropout_rate,
                    training=self.training,
                )

        x_site_int = node_emb_dict["site_int"]

        if self.fc1 is not None:
            x_site_int = mlt.torch.get_act_fn(self.run_config.fc_act_fn)(
                self.fc1(x_site_int)
            )

        out = self.out_fc(x_site_int)
        ln_im_mean, ln_im_std = out.chunk(2, dim=1)

        # Clip predicted values to prevent numerical issues
        ln_im_std = torch.clamp(ln_im_std, min=-3, max=1)
        ln_im_mean = torch.clamp(ln_im_mean, min=-3, max=1)

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
        int_update_model: nn.Module,
        edge_update_model: nn.Module,
        att_model: nn.Module,
        **kwargs,
    ):
        """
        Parameters
        ----------
        obs_transform_model: torch.nn.Module
            Transformation model for the observation nodes
        int_update_model: torch.nn.Module
            Update model for the target nodes
        edge_update_model: torch.nn.Module
            Update model for the edges
        att_model: torch.nn.Module
            Self-attention model
        kwargs
        """
        super().__init__(aggr="add", **kwargs)

        self.att_model = att_model

        self.obs_transform_models = obs_transform_models
        self.int_update_model = int_update_model
        self.edge_update_model = edge_update_model

        self.n_heads = len(self.obs_transform_models)

        # self.obs_transform_model = obs_transform_model
        # self.int_transform_model = int_transform_model

        # self.aggr = "add"
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        ginits.reset(self.att_model)
        ginits.reset(self.obs_transform_models)
        ginits.reset(self.int_update_model)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        edge_attr = self.edge_update(edge_attr)

        # Compute the messages
        m_s = self.propagate(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
        )

        # Update the nodes
        out = m_s + self.int_update_model(x[1])
        return out, edge_attr

    def edge_update(self, edge_attr: torch.Tensor) -> Tensor:
        edge_attr = self.edge_update_model(edge_attr)
        return edge_attr

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


class AddSoIObsIMsTransform(BaseTransform):
    """
    A transform for adding an observation node for the SoI 
    to the graph data with a certain probability.
    This forces to model to learn to use the observation 
    at the SoI if they are available, additionally it helps
    teach the model site-to-site correlation 
    (i.e. very close sites are highly correlated).
    """


    def __init__(
        self,
        event_scalar_feature_dfs: dict[str, pd.DataFrame],
        run_config: "gnn_gm.RunConfig",
    ):
        super().__init__()

        self.event_scalar_feature_dfs = event_scalar_feature_dfs
        self.pert_probability = run_config.soi_with_obs_pert_prob

        self.run_config = run_config

        # Set the seed
        random.seed(run_config.seed)

    def forward(self, data: gdata.HeteroData) -> gdata.HeteroData:
        if random.uniform(0, 1) < self.pert_probability:
            if self.run_config.graph_feature_keys["site_obs"] is None:
                event = data["metadata"]["event"]
                site_int = data["metadata"]["site_int"]
                n_obs_sites = data["metadata"]["obs_sites"].size
                scalar_feature_df = self.event_scalar_feature_dfs[event]

                data["site_obs"]["x"] = torch.cat(
                    (data["site_obs"]["x"], torch.nan_to_num(data["y"], nan=99)), dim=0
                )
                data[("site_obs", "self_loop", "site_obs")]["edge_index"] = torch.arange(
                    0, n_obs_sites + 1
                ).tile((2, 1))

                data[("site_obs", "informs", "site_int")]["edge_index"] = torch.cat(
                    (
                        torch.arange(0, n_obs_sites + 1)[None, :],
                        torch.zeros((1, n_obs_sites + 1), dtype=int),
                    ),
                    dim=0,
                )
                data[("site_obs", "informs", "site_int")]["edge_attr"] = torch.cat(
                    (
                        data[("site_obs", "informs", "site_int")]["edge_attr"],
                        torch.from_numpy(scalar_feature_df.loc[f"{site_int}_{site_int}", self.run_config.graph_feature_keys["edge"]].values).to(torch.float32)[None, :]
                    ),
                    dim=0,
                )

                return data
            else:
                raise NotImplementedError()
        else:
            return data


class CustomGraphDataset(gdata.InMemoryDataset):

    def __init__(self, graph_objects: list[gdata.HeteroData], transform=None):
        super().__init__(".", transform=transform)

        self.data, self.slices = self.collate(graph_objects)

    def _download(self):
        pass

    def _process(self):
        pass
