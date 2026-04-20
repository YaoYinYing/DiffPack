from __future__ import annotations

import os
import pprint
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diffpack import repack, rotamer, util
from diffpack.device import choose_torch_device, move_to_device
from diffpack.schedule_cache import (
    load_schedule_tables_readonly,
    resolve_cache_root,
    validate_required_schedule_caches,
)
try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional at runtime
    Chem = None

try:
    from torch_cluster import knn_graph as _knn_graph_impl, radius_graph as _radius_graph_impl
except Exception:  # pragma: no cover - runtime optional acceleration
    _knn_graph_impl = None
    _radius_graph_impl = None


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = _scatter_add(src, index, dim_size)
    count = torch.bincount(index, minlength=dim_size).to(src.device)
    count = count.clamp_min(1).to(src.dtype)
    if src.dim() > 1:
        count = count.unsqueeze(-1)
    return out / count


def _pairwise_edges(position: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    src_list = []
    dst_list = []
    for graph_id in batch.unique(sorted=True):
        nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        if nodes.numel() == 0:
            continue
        src = nodes.repeat_interleave(nodes.numel())
        dst = nodes.repeat(nodes.numel())
        mask = src != dst
        src_list.append(src[mask])
        dst_list.append(dst[mask])
    if not src_list:
        empty = torch.zeros((0,), dtype=torch.long, device=position.device)
        return empty, empty
    return torch.cat(src_list), torch.cat(dst_list)


def _knn_graph(position: torch.Tensor, k: int, batch: torch.Tensor) -> torch.Tensor:
    if _knn_graph_impl is not None:
        return _knn_graph_impl(position, k=k, batch=batch, loop=False)

    src, dst = _pairwise_edges(position, batch)
    if src.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=position.device)
    dist = (position[src] - position[dst]).norm(dim=-1)
    edge_src = []
    edge_dst = []
    for node in range(position.shape[0]):
        mask = dst == node
        if mask.sum() == 0:
            continue
        local_src = src[mask]
        local_dist = dist[mask]
        topk = min(k, local_src.numel())
        idx = torch.argsort(local_dist)[:topk]
        edge_src.append(local_src[idx])
        edge_dst.append(torch.full((topk,), node, dtype=torch.long, device=position.device))
    if not edge_src:
        return torch.zeros((2, 0), dtype=torch.long, device=position.device)
    return torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)


def _radius_graph(position: torch.Tensor, radius: float, batch: torch.Tensor, max_num_neighbors: int) -> torch.Tensor:
    if _radius_graph_impl is not None:
        return _radius_graph_impl(
            position,
            r=radius,
            batch=batch,
            max_num_neighbors=max_num_neighbors,
            loop=False,
        )

    src, dst = _pairwise_edges(position, batch)
    if src.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=position.device)
    dist = (position[src] - position[dst]).norm(dim=-1)
    mask = dist <= radius
    src, dst, dist = src[mask], dst[mask], dist[mask]
    if src.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=position.device)

    edge_src = []
    edge_dst = []
    for node in range(position.shape[0]):
        node_mask = dst == node
        if node_mask.sum() == 0:
            continue
        local_src = src[node_mask]
        local_dist = dist[node_mask]
        if local_src.numel() > max_num_neighbors:
            idx = torch.argsort(local_dist)[:max_num_neighbors]
            local_src = local_src[idx]
        edge_src.append(local_src)
        edge_dst.append(torch.full((local_src.numel(),), node, dtype=torch.long, device=position.device))
    if not edge_src:
        return torch.zeros((2, 0), dtype=torch.long, device=position.device)
    return torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)


@dataclass
class PygProteinGraph:
    protein: Any
    edge_list: torch.Tensor
    edge_feature: torch.Tensor | None
    num_relation: int
    edge_weight: torch.Tensor | None = None

    @property
    def device(self):
        return self.protein.device

    @property
    def num_node(self):
        return int(self.protein.num_node)

    @property
    def num_edge(self):
        return int(self.edge_list.shape[0])

    @property
    def batch_size(self):
        return int(self.protein.batch_size)

    @property
    def node_feature(self):
        return self.protein.node_feature

    @property
    def node_position(self):
        return self.protein.node_position

    @property
    def atom2graph(self):
        return self.protein.atom2graph

    @property
    def atom2residue(self):
        return self.protein.atom2residue

    @property
    def residue2graph(self):
        return self.protein.residue2graph

    @property
    def num_residue(self):
        return int(self.protein.num_residue)

    @property
    def residue_type(self):
        return self.protein.residue_type

    @property
    def chi_mask(self):
        return self.protein.chi_mask

    @property
    def chi_1pi_periodic_mask(self):
        return self.protein.chi_1pi_periodic_mask

    @property
    def chi_2pi_periodic_mask(self):
        return self.protein.chi_2pi_periodic_mask


class PygBondEdge(nn.Module):
    def forward(self, protein):
        edge_list = protein.edge_list
        if edge_list.shape[1] == 2:
            relation = torch.zeros((edge_list.shape[0], 1), dtype=torch.long, device=protein.device)
            edge_list = torch.cat([edge_list, relation], dim=-1)
            return edge_list, 1
        num_relation = int(getattr(protein, "num_relation", 0))
        if num_relation <= 0:
            num_relation = int(edge_list[:, 2].max().item()) + 1 if edge_list.numel() else 1
        return edge_list, num_relation


class PygKNNEdge(nn.Module):
    eps = 1e-10

    def __init__(self, k=10, min_distance=5, max_distance=None):
        super().__init__()
        self.k = int(k)
        self.min_distance = int(min_distance)
        self.max_distance = max_distance

    def forward(self, protein):
        edge_index = _knn_graph(protein.node_position, self.k, protein.atom2graph)
        edge_list = edge_index.t().contiguous()
        relation = torch.zeros((edge_list.shape[0], 1), dtype=torch.long, device=protein.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        if self.min_distance > 0:
            node_in, node_out = edge_list[:, 0], edge_list[:, 1]
            mask = (protein.atom2residue[node_in] - protein.atom2residue[node_out]).abs() >= self.min_distance
            edge_list = edge_list[mask]
        if self.max_distance is not None:
            node_in, node_out = edge_list[:, 0], edge_list[:, 1]
            mask = (protein.atom2residue[node_in] - protein.atom2residue[node_out]).abs() <= int(self.max_distance)
            edge_list = edge_list[mask]
        if edge_list.numel():
            node_in, node_out = edge_list[:, 0], edge_list[:, 1]
            dist = (protein.node_position[node_in] - protein.node_position[node_out]).norm(dim=-1)
            edge_list = edge_list[dist >= self.eps]
        return edge_list, 1


class PygSpatialEdge(nn.Module):
    eps = 1e-10

    def __init__(self, radius=5.0, min_distance=5, max_distance=None, max_num_neighbors=32):
        super().__init__()
        self.radius = float(radius)
        self.min_distance = int(min_distance)
        self.max_distance = max_distance
        self.max_num_neighbors = int(max_num_neighbors)

    def forward(self, protein):
        edge_index = _radius_graph(protein.node_position, self.radius, protein.atom2graph, self.max_num_neighbors)
        edge_list = edge_index.t().contiguous()
        relation = torch.zeros((edge_list.shape[0], 1), dtype=torch.long, device=protein.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        if self.min_distance > 0:
            node_in, node_out = edge_list[:, 0], edge_list[:, 1]
            mask = (protein.atom2residue[node_in] - protein.atom2residue[node_out]).abs() >= self.min_distance
            edge_list = edge_list[mask]
        if self.max_distance is not None:
            node_in, node_out = edge_list[:, 0], edge_list[:, 1]
            mask = (protein.atom2residue[node_in] - protein.atom2residue[node_out]).abs() <= int(self.max_distance)
            edge_list = edge_list[mask]
        if edge_list.numel():
            node_in, node_out = edge_list[:, 0], edge_list[:, 1]
            dist = (protein.node_position[node_in] - protein.node_position[node_out]).norm(dim=-1)
            edge_list = edge_list[dist >= self.eps]
        return edge_list, 1


class PygGraphConstruction(nn.Module):
    max_seq_dist = 10

    def __init__(self, edge_layers: list[nn.Module], edge_feature: str | None = "gearnet"):
        super().__init__()
        self.edge_layers = nn.ModuleList(edge_layers)
        self.edge_feature = edge_feature

    def _edge_gearnet(self, protein, edge_list: torch.Tensor, num_relation: int):
        node_in, node_out, relation = edge_list.t()
        residue_in = protein.atom2residue[node_in]
        residue_out = protein.atom2residue[node_out]
        in_residue_type = protein.residue_type[residue_in]
        out_residue_type = protein.residue_type[residue_out]
        sequential_dist = torch.abs(residue_in - residue_out).clamp(max=self.max_seq_dist)
        spatial_dist = (protein.node_position[node_in] - protein.node_position[node_out]).norm(dim=-1)
        return torch.cat(
            [
                F.one_hot(in_residue_type, num_classes=21).to(torch.float32),
                F.one_hot(out_residue_type, num_classes=21).to(torch.float32),
                F.one_hot(relation, num_classes=num_relation).to(torch.float32),
                F.one_hot(sequential_dist, num_classes=self.max_seq_dist + 1).to(torch.float32),
                spatial_dist.unsqueeze(-1),
            ],
            dim=-1,
        )

    def forward(self, protein):
        if not self.edge_layers:
            edge_list = protein.edge_list
            if edge_list.shape[1] == 2:
                relation = torch.zeros((edge_list.shape[0], 1), dtype=torch.long, device=protein.device)
                edge_list = torch.cat([edge_list, relation], dim=-1)
                num_relation = 1
            else:
                num_relation = int(getattr(protein, "num_relation", 0))
                if num_relation <= 0:
                    num_relation = int(edge_list[:, 2].max().item()) + 1 if edge_list.numel() else 1
            edge_feature = None
            if self.edge_feature == "gearnet":
                edge_feature = self._edge_gearnet(protein, edge_list, num_relation)
            edge_weight = torch.ones(edge_list.shape[0], dtype=torch.float32, device=protein.device)
            return PygProteinGraph(
                protein=protein,
                edge_list=edge_list,
                edge_feature=edge_feature,
                num_relation=num_relation,
                edge_weight=edge_weight,
            )

        parts = []
        rel_counts = []
        for layer in self.edge_layers:
            edges, nrel = layer(protein)
            parts.append(edges)
            rel_counts.append(int(nrel))
        relation_offset = 0
        edge_list = []
        for edges, nrel in zip(parts, rel_counts):
            if edges.numel():
                shifted = edges.clone()
                shifted[:, 2] += relation_offset
                edge_list.append(shifted)
            relation_offset += int(nrel)
        if edge_list:
            edge_list = torch.cat(edge_list, dim=0)
            num_relation = int(relation_offset)
        elif relation_offset > 0:
            edge_list = torch.zeros((0, 3), dtype=torch.long, device=protein.device)
            num_relation = int(relation_offset)
        else:
            edge_list = protein.edge_list
            if edge_list.shape[1] == 2:
                relation = torch.zeros((edge_list.shape[0], 1), dtype=torch.long, device=protein.device)
                edge_list = torch.cat([edge_list, relation], dim=-1)
                num_relation = 1
            else:
                num_relation = int(edge_list[:, 2].max().item()) + 1 if edge_list.numel() else 1

        edge_feature = None
        if self.edge_feature == "gearnet":
            edge_feature = self._edge_gearnet(protein, edge_list, num_relation)
        elif self.edge_feature is not None:
            raise ValueError(f"Unsupported edge_feature for PyG backend: {self.edge_feature}")

        edge_weight = torch.ones(edge_list.shape[0], dtype=torch.float32, device=protein.device)
        return PygProteinGraph(
            protein=protein,
            edge_list=edge_list,
            edge_feature=edge_feature,
            num_relation=num_relation,
            edge_weight=edge_weight,
        )


class PygRelationalGraphConv(nn.Module):
    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_relation = int(num_relation)
        self.edge_input_dim = edge_input_dim
        self.self_loop = nn.Linear(self.input_dim, self.output_dim)
        self.linear = nn.Linear(self.num_relation * self.input_dim, self.output_dim)
        self.edge_linear = nn.Linear(edge_input_dim, self.input_dim) if edge_input_dim else None
        self.batch_norm = nn.BatchNorm1d(self.output_dim) if batch_norm else None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def aggregate(self, graph: PygProteinGraph, message: torch.Tensor):
        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        if graph.edge_weight is None:
            edge_weight = torch.ones(graph.num_edge, device=graph.device, dtype=message.dtype)
        else:
            edge_weight = graph.edge_weight.to(message.dtype)
        weighted = message * edge_weight.unsqueeze(-1)
        denom = _scatter_add(edge_weight.unsqueeze(-1), node_out, graph.num_node * self.num_relation)
        update = _scatter_add(weighted, node_out, graph.num_node * self.num_relation) / (denom + self.eps)
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def forward(self, graph: PygProteinGraph, input_feature: torch.Tensor):
        if graph.num_relation != self.num_relation:
            raise ValueError(f"Relation mismatch: graph={graph.num_relation} layer={self.num_relation}")
        node_in = graph.edge_list[:, 0]
        message = input_feature[node_in]
        if self.edge_linear is not None and graph.edge_feature is not None:
            message = message + self.edge_linear(graph.edge_feature.float())
        update = self.aggregate(graph, message)
        output = self.linear(update) + self.self_loop(input_feature)
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class PygGeometricRelationalGraphConv(PygRelationalGraphConv):
    def aggregate(self, graph: PygProteinGraph, message: torch.Tensor):
        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        if graph.edge_weight is None:
            weighted = message
        else:
            weighted = message * graph.edge_weight.to(message.dtype).unsqueeze(-1)
        update = _scatter_add(weighted, node_out, graph.num_node * self.num_relation)
        return update.view(graph.num_node, self.num_relation * self.input_dim)


@dataclass
class PygLineGraph:
    edge_list: torch.Tensor
    node_feature: torch.Tensor
    num_relation: int
    num_node: int
    device: torch.device
    edge_feature: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None


class PygSpatialLineGraph(nn.Module):
    def __init__(self, num_angle_bin=8):
        super().__init__()
        self.num_angle_bin = int(num_angle_bin)

    def forward(self, graph: PygProteinGraph):
        edge_index = graph.edge_list[:, :2]
        num_edge = edge_index.shape[0]
        if num_edge == 0:
            empty_edge = torch.zeros((0, 3), dtype=torch.long, device=graph.device)
            node_feature = graph.edge_feature if graph.edge_feature is not None else torch.zeros((0, 0), device=graph.device)
            return PygLineGraph(empty_edge, node_feature, self.num_angle_bin, 0, graph.device, edge_weight=torch.zeros(0, device=graph.device))

        node_in = edge_index[:, 0]
        node_out = edge_index[:, 1]
        pairs = []
        for middle in range(graph.num_node):
            incoming = torch.nonzero(node_out == middle, as_tuple=False).flatten()
            outgoing = torch.nonzero(node_in == middle, as_tuple=False).flatten()
            if incoming.numel() == 0 or outgoing.numel() == 0:
                continue
            edge_in = incoming.repeat_interleave(outgoing.numel())
            edge_out = outgoing.repeat(incoming.numel())
            pairs.append(torch.stack([edge_in, edge_out], dim=-1))
        if pairs:
            lg = torch.cat(pairs, dim=0)
        else:
            lg = torch.zeros((0, 2), dtype=torch.long, device=graph.device)

        if lg.numel():
            edge_in, edge_out = lg[:, 0], lg[:, 1]
            node_i = node_out[edge_out]
            node_j = node_in[edge_out]
            node_k = node_in[edge_in]
            vector1 = graph.node_position[node_i] - graph.node_position[node_j]
            vector2 = graph.node_position[node_k] - graph.node_position[node_j]
            x = (vector1 * vector2).sum(dim=-1)
            y = torch.cross(vector1, vector2, dim=-1).norm(dim=-1)
            angle = torch.atan2(y, x)
            relation = (angle / np.pi * self.num_angle_bin).long().clamp(max=self.num_angle_bin - 1)
            edge_list = torch.cat([lg, relation.unsqueeze(-1)], dim=-1)
        else:
            edge_list = torch.zeros((0, 3), dtype=torch.long, device=graph.device)

        node_feature = graph.edge_feature if graph.edge_feature is not None else torch.zeros((num_edge, 0), device=graph.device)
        edge_weight = torch.ones(edge_list.shape[0], dtype=torch.float32, device=graph.device)
        return PygLineGraph(
            edge_list=edge_list,
            node_feature=node_feature,
            num_relation=self.num_angle_bin,
            num_node=num_edge,
            device=graph.device,
            edge_weight=edge_weight,
        )


class PygGearNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_relation,
        edge_input_dim=None,
        num_angle_bin=None,
        short_cut=False,
        batch_norm=False,
        activation="relu",
        concat_hidden=False,
        readout="sum",
    ):
        super().__init__()
        if not isinstance(hidden_dims, (list, tuple)):
            hidden_dims = [hidden_dims]
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.num_relation = int(num_relation)
        self.short_cut = bool(short_cut)
        self.concat_hidden = bool(concat_hidden)
        self.readout = readout
        self.use_post_batch_norm = bool(batch_norm)

        dims = [int(input_dim)] + [int(d) for d in hidden_dims]
        edge_dims = [int(edge_input_dim)] + dims[:-1]
        self.layers = nn.ModuleList(
            [
                PygGeometricRelationalGraphConv(dims[i], dims[i + 1], self.num_relation, None, batch_norm, activation)
                for i in range(len(dims) - 1)
            ]
        )
        if self.use_post_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)])
        self.num_angle_bin = int(num_angle_bin) if num_angle_bin else None
        if self.num_angle_bin:
            self.spatial_line_graph = PygSpatialLineGraph(self.num_angle_bin)
            self.edge_layers = nn.ModuleList(
                [
                    PygGeometricRelationalGraphConv(edge_dims[i], edge_dims[i + 1], self.num_angle_bin, None, batch_norm, activation)
                    for i in range(len(edge_dims) - 1)
                ]
            )

    def forward(self, graph: PygProteinGraph, input_feature: torch.Tensor, all_loss=None, metric=None):
        hiddens = []
        layer_input = input_feature
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()
            expected_edge_dim = self.edge_layers[0].input_dim
            if edge_input.shape[1] != expected_edge_dim:
                if edge_input.shape[1] > expected_edge_dim:
                    edge_input = edge_input[:, :expected_edge_dim]
                else:
                    pad = edge_input.new_zeros((edge_input.shape[0], expected_edge_dim - edge_input.shape[1]))
                    edge_input = torch.cat([edge_input, pad], dim=-1)

        for i, conv in enumerate(self.layers):
            hidden = conv(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                if graph.edge_weight is None:
                    weighted_edge_hidden = edge_hidden
                else:
                    weighted_edge_hidden = edge_hidden * graph.edge_weight.to(edge_hidden.dtype).unsqueeze(-1)
                update = _scatter_add(weighted_edge_hidden, node_out, graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = conv.linear(update)
                if conv.activation is not None:
                    update = conv.activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.use_post_batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        node_feature = torch.cat(hiddens, dim=-1) if self.concat_hidden else hiddens[-1]
        if graph.batch_size == 1:
            graph_feature = node_feature.sum(dim=0, keepdim=True) if self.readout == "sum" else node_feature.mean(dim=0, keepdim=True)
        else:
            graph_ids = graph.atom2graph
            if self.readout == "sum":
                graph_feature = _scatter_add(node_feature, graph_ids, graph.batch_size)
            else:
                graph_feature = _scatter_mean(node_feature, graph_ids, graph.batch_size)
        return {"graph_feature": graph_feature, "node_feature": node_feature}


class PygMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], short_cut: bool = False):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.short_cut = short_cut

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            y = layer(x)
            if i != len(self.layers) - 1:
                y = F.relu(y)
            if self.short_cut and y.shape == x.shape:
                y = y + x
            x = y
        return x


class PygSigmaEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, sigma_dim, embed_type="sinusoidal", operation="post_add"):
        super().__init__()
        self.output_dim = hidden_dims[-1]
        self.embed_type = embed_type
        self.sigma_dim = sigma_dim
        self.operation = operation
        if self.operation == "post_add":
            self.sigma_linear = nn.Linear(sigma_dim, hidden_dims[-1])
            self.mlp = PygMLP(input_dim, hidden_dims, short_cut=True)
        elif self.operation == "pre_concat":
            self.mlp = PygMLP(input_dim + sigma_dim, hidden_dims, short_cut=True)
        else:
            raise ValueError(f"Unsupported sigma embedding operation `{operation}`")

    def _embed_sigma(self, sigma: torch.Tensor):
        if sigma.ndim != 1:
            sigma = sigma.flatten()
        half_dim = self.sigma_dim // 2
        scale = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=sigma.device) * -(np.log(10000) / max(half_dim - 1, 1)))
        emb = sigma.float().unsqueeze(-1) * scale.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.sigma_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, input_feature, sigma):
        sigma_embed = self._embed_sigma(sigma)
        if self.operation == "post_add":
            return self.mlp(input_feature) + self.sigma_linear(sigma_embed)
        return self.mlp(torch.cat([input_feature, sigma_embed], dim=-1))


def _periodic_p(x: np.ndarray, sigma: np.ndarray, N: int = 10, PI: float = np.pi):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += np.exp(-((x + 2 * PI * i) ** 2) / (2 * sigma ** 2))
    return p_


def _periodic_grad(x: np.ndarray, sigma: np.ndarray, N: int = 10, PI: float = np.pi):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + 2 * PI * i) / (sigma ** 2) * np.exp(-((x + 2 * PI * i) ** 2) / (2 * sigma ** 2))
    return p_


def _periodic_sample(sigma: np.ndarray, PI: float = np.pi):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + PI) % (2 * PI) - PI
    return out


class PygSO2Schedule(nn.Module):
    X_MIN, X_N = 1e-5, 5000
    SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000

    def __init__(self, PI: float, cache_folder: str | None, cache_read_only: bool = True):
        super().__init__()
        self.PI = PI
        self.cache_folder = resolve_cache_root(cache_folder)
        if not cache_read_only:
            raise RuntimeError(
                "Inference cache is read-only. Use `diffpack-prepare-cache` to build/repair caches before inference."
            )
        self.p_, self.score_, self.score_norm_ = load_schedule_tables_readonly(self.cache_folder, PI)

    def score(self, x, sigma):
        x = (x + self.PI) % (2 * self.PI) - self.PI
        sign = np.sign(x)
        x = np.log(np.abs(x) / self.PI + 1e-10)
        x = (x - np.log(self.X_MIN)) / (0 - np.log(self.X_MIN)) * self.X_N
        x = np.round(np.clip(x, 0, self.X_N)).astype(int)
        sigma = np.log(sigma / self.PI)
        sigma = (sigma - np.log(self.SIGMA_MIN)) / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN)) * self.SIGMA_N
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return -sign * self.score_[sigma, x]

    def p(self, x, sigma):
        x = (x + self.PI) % (2 * self.PI) - self.PI
        x = np.log(np.abs(x) / self.PI + 1e-10)
        x = (x - np.log(self.X_MIN)) / (0 - np.log(self.X_MIN)) * self.X_N
        x = np.round(np.clip(x, 0, self.X_N)).astype(int)
        sigma = np.log(sigma / self.PI)
        sigma = (sigma - np.log(self.SIGMA_MIN)) / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN)) * self.SIGMA_N
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return self.p_[sigma, x]

    def score_norm(self, sigma):
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.detach().cpu().numpy()
        sigma = np.log(sigma / self.PI)
        sigma = (sigma - np.log(self.SIGMA_MIN)) / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN)) * self.SIGMA_N
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return self.score_norm_[sigma]


class PygSO2VESchedule(PygSO2Schedule):
    def __init__(
        self,
        pi_periodic=False,
        cache_folder=None,
        cache_read_only=True,
        sigma_min=0.01 * np.pi,
        sigma_max=np.pi,
        annealed_temp=3,
        mode="sde",
        **kwargs,
    ):
        PI = (0.5 * np.pi) if pi_periodic else np.pi
        super().__init__(PI=PI, cache_folder=cache_folder, cache_read_only=cache_read_only)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_min_log = np.log(sigma_min)
        self.sigma_max_log = np.log(sigma_max)
        self.annealed_temp = annealed_temp
        self.mode = mode
        self._reverse_t_schedule = torch.linspace(1.0, 0.0, 11)

    @property
    def reverse_t_schedule(self):
        return self._reverse_t_schedule

    def sample_train_t(self, shape):
        return torch.rand(shape)

    def t_to_sigma(self, t):
        return torch.exp(self.sigma_min_log + (self.sigma_max_log - self.sigma_min_log) * t)

    def score(self, x, sigma):
        return super().score(x, sigma)

    def score_norm(self, sigma):
        return super().score_norm(sigma)

    @torch.no_grad()
    def add_noise(self, x, t, x_mask=None):
        t = t.to(dtype=x.dtype)
        sigmas = self.t_to_sigma(t).to(dtype=x.dtype)
        noise = torch.randn_like(x) * sigmas.unsqueeze(-1)
        score = torch.tensor(self.score(noise.cpu().numpy(), sigmas.cpu().numpy()), device=x.device, dtype=x.dtype)
        if x_mask is not None:
            noise = noise * x_mask
            score = score * x_mask
        return x + noise, score

    @torch.no_grad()
    def step(self, x, x_score, t, dt, x_mask=None):
        t = t.to(dtype=x.dtype)
        dt = dt.to(dtype=x.dtype)
        sigma = self.t_to_sigma(t).to(dtype=x.dtype)
        g = sigma * float(np.sqrt(2 * np.log(self.sigma_max / self.sigma_min)))
        alpha = 1 - (sigma / float(np.exp(self.sigma_max_log))) ** 2 if self.annealed_temp else None
        annealed_weight = self.annealed_temp / (alpha + (1 - alpha) * self.annealed_temp) if self.annealed_temp else 1
        if not isinstance(annealed_weight, torch.Tensor):
            annealed_weight = torch.tensor(float(annealed_weight), device=x.device, dtype=x.dtype)
        else:
            annealed_weight = annealed_weight.to(dtype=x.dtype)
        if self.mode == "ode":
            x_prev = x + 0.5 * g ** 2 * dt * (x_score * annealed_weight)
        elif self.mode == "sde":
            noise = torch.randn_like(x_score)
            x_prev = x + g ** 2 * dt * (x_score * annealed_weight) + g * torch.sqrt(dt) * noise
        else:
            raise NotImplementedError(f"Unknown schedule mode `{self.mode}`")
        x_prev = x_prev.to(dtype=x.dtype)
        if x_mask is not None:
            x_prev[~x_mask] = x[~x_mask]
        return x_prev


class PygProtein:
    def __init__(
        self,
        node_position: torch.Tensor,
        atom_name: torch.Tensor,
        atom2residue: torch.Tensor,
        residue_type: torch.Tensor,
        residue_chain: list[str],
        residue_number: list[int],
        edge_list: torch.Tensor,
        num_relation: int,
        node_feature: torch.Tensor,
    ):
        self.node_position = node_position
        self.atom_name = atom_name
        self.atom2residue = atom2residue
        self.residue_type = residue_type
        self.residue_chain = list(residue_chain)
        self.residue_number = list(residue_number)
        self.edge_list = edge_list
        self.num_relation = int(num_relation)
        self.node_feature = node_feature
        self.atom2graph = torch.zeros(node_position.shape[0], dtype=torch.long, device=node_position.device)
        self.residue2graph = torch.zeros(residue_type.shape[0], dtype=torch.long, device=node_position.device)
        self.batch_size = 1
        self.atom14index = rotamer.restype_atom14_index_map.to(node_position.device)[
            self.residue_type[self.atom2residue], self.atom_name
        ]
        self._refresh_masks()

    @property
    def device(self):
        return self.node_position.device

    @property
    def num_node(self):
        return self.node_position.shape[0]

    @property
    def num_residue(self):
        return self.residue_type.shape[0]

    def _refresh_masks(self):
        with torch.no_grad():
            chi_mask = rotamer.get_chi_mask(self)
            chi_1pi_periodic_mask = torch.tensor(rotamer.chi_pi_periodic, device=self.device)[self.residue_type]
            chi_2pi_periodic_mask = ~chi_1pi_periodic_mask
            self.chi_mask = chi_mask
            self.chi_1pi_periodic_mask = chi_mask & chi_1pi_periodic_mask
            self.chi_2pi_periodic_mask = chi_mask & chi_2pi_periodic_mask
            self.atom37_mask = torch.zeros(self.num_residue, len(rotamer.atom_name_vocab), device=self.device, dtype=torch.bool)
            self.atom37_mask[self.atom2residue, self.atom_name] = True
            self.sidechain37_mask = self.atom37_mask.clone()
            self.sidechain37_mask[:, rotamer.bb_atom_name] = False
            self.repack_residue_mask = torch.ones(self.num_residue, device=self.device, dtype=torch.bool)

    def clone(self):
        out = PygProtein(
            node_position=self.node_position.clone(),
            atom_name=self.atom_name.clone(),
            atom2residue=self.atom2residue.clone(),
            residue_type=self.residue_type.clone(),
            residue_chain=list(self.residue_chain),
            residue_number=list(self.residue_number),
            edge_list=self.edge_list.clone(),
            num_relation=self.num_relation,
            node_feature=self.node_feature.clone(),
        )
        out.chi_mask = self.chi_mask.clone()
        out.chi_1pi_periodic_mask = self.chi_1pi_periodic_mask.clone()
        out.chi_2pi_periodic_mask = self.chi_2pi_periodic_mask.clone()
        out.atom37_mask = self.atom37_mask.clone()
        out.sidechain37_mask = self.sidechain37_mask.clone()
        out.repack_residue_mask = self.repack_residue_mask.clone()
        return out

    def to(self, device):
        out = self.clone()
        for name in [
            "node_position", "atom_name", "atom2residue", "residue_type", "edge_list", "node_feature", "atom2graph",
            "residue2graph", "atom14index", "chi_mask", "chi_1pi_periodic_mask", "chi_2pi_periodic_mask",
            "atom37_mask", "sidechain37_mask", "repack_residue_mask",
        ]:
            setattr(out, name, getattr(out, name).to(device))
        return out

    def cpu(self):
        return self.to(torch.device("cpu"))

    def subgraph(self, keep_atom_mask: torch.Tensor):
        keep_atom_mask = keep_atom_mask.bool()
        old_to_new_atom = torch.full((self.num_node,), -1, dtype=torch.long, device=self.device)
        kept_atoms = torch.nonzero(keep_atom_mask, as_tuple=False).flatten()
        old_to_new_atom[kept_atoms] = torch.arange(kept_atoms.numel(), device=self.device)
        new_atom2res_old = self.atom2residue[keep_atom_mask]
        kept_res_old = []
        seen = set()
        for rid in new_atom2res_old.tolist():
            if rid not in seen:
                seen.add(rid)
                kept_res_old.append(rid)
        old_to_new_res = torch.full((self.num_residue,), -1, dtype=torch.long, device=self.device)
        old_to_new_res[torch.tensor(kept_res_old, device=self.device)] = torch.arange(len(kept_res_old), device=self.device)
        new_atom2res = old_to_new_res[new_atom2res_old]
        edge_keep = keep_atom_mask[self.edge_list[:, 0]] & keep_atom_mask[self.edge_list[:, 1]]
        new_edge = self.edge_list[edge_keep].clone()
        if new_edge.numel():
            new_edge[:, 0] = old_to_new_atom[new_edge[:, 0]]
            new_edge[:, 1] = old_to_new_atom[new_edge[:, 1]]
        out = PygProtein(
            node_position=self.node_position[keep_atom_mask],
            atom_name=self.atom_name[keep_atom_mask],
            atom2residue=new_atom2res,
            residue_type=self.residue_type[torch.tensor(kept_res_old, device=self.device)],
            residue_chain=[self.residue_chain[i] for i in kept_res_old],
            residue_number=[self.residue_number[i] for i in kept_res_old],
            edge_list=new_edge,
            num_relation=self.num_relation,
            node_feature=self.node_feature[keep_atom_mask],
        )
        if hasattr(self, "repack_residue_mask"):
            out.repack_residue_mask = self.repack_residue_mask[torch.tensor(kept_res_old, device=self.device)]
        return out

    def to_pdb(self, path: str):
        inv_atom_name = {v: k for k, v in rotamer.atom_name_vocab.items()}
        residue_names = rotamer.residue_list
        lines = []
        serial = 1
        for i in range(self.num_node):
            res_id = int(self.atom2residue[i].item())
            atom_name = inv_atom_name[int(self.atom_name[i].item())]
            res_name = residue_names[int(self.residue_type[res_id].item())]
            chain = self.residue_chain[res_id] or "A"
            res_num = int(self.residue_number[res_id])
            x, y, z = self.node_position[i].tolist()
            elem = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:>4s} {res_name:>3s} {chain:1s}{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {elem:>2s}"
            )
            serial += 1
        lines.extend(["TER", "END"])
        Path(path).write_text("\n".join(lines), encoding="utf-8")


class PygSideChainDataset:
    _atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
    _atom_vocab_map = {name: i for i, name in enumerate(_atom_vocab)}
    _residue_vocab = ["GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
                     "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"]
    _residue_vocab_map = {name: i for i, name in enumerate(_residue_vocab)}

    def __init__(self, pdb_files=None, center_residues=None, repack_radius=None, hetero_policy="exclude", **kwargs):
        self.pdb_files = [os.path.expanduser(p) for p in (pdb_files or [])]
        self.center_residue_selectors = repack.parse_center_residue_selectors(center_residues or [])
        self.repack_radius = repack_radius
        self.hetero_policy = hetero_policy

    def __len__(self):
        return len(self.pdb_files)

    def _parse_pdb(self, path: str):
        atom_name = []
        atom_symbol = []
        residue_name = []
        atom_pos = []
        residue_keys = []
        residue_type = []
        atom2res = []
        atom_keys = []
        residue_index = {}
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if self.hetero_policy == "error" and any(line.startswith("HETATM") for line in lines):
            raise ValueError(f"HETATM records found in `{path}` while `hetero_policy=error`.")
        for line in lines:
            if not line.startswith("ATOM"):
                continue
            aname = line[12:16].strip()
            rname = line[17:20].strip()
            chain = (line[21] or "A").strip() or "A"
            try:
                rnum = int(line[22:26].strip())
            except ValueError:
                continue
            if rname not in rotamer.residue_vocab:
                continue
            if aname not in rotamer.atom_name_vocab:
                continue
            key = (chain, rnum)
            if key not in residue_index:
                residue_index[key] = len(residue_keys)
                residue_keys.append(key)
                residue_type.append(rotamer.residue_vocab[rname])
            rid = residue_index[key]
            atom2res.append(rid)
            atom_name.append(rotamer.atom_name_vocab[aname])
            element = line[76:78].strip()
            if not element:
                alpha = "".join(ch for ch in aname if ch.isalpha())
                element = alpha[:1] if alpha else "C"
            atom_symbol.append(element)
            residue_name.append(rname)
            atom_pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            atom_keys.append((chain, rnum, aname))
        if len(atom_name) == 0:
            raise ValueError(f"No valid ATOM records parsed from `{path}`.")
        atom_name_t = torch.tensor(atom_name, dtype=torch.long)
        atom_pos_t = torch.tensor(atom_pos, dtype=torch.float32)
        atom2res_t = torch.tensor(atom2res, dtype=torch.long)
        residue_type_t = torch.tensor(residue_type, dtype=torch.long)
        node_feature = torch.zeros((len(atom_name), 39), dtype=torch.float32)
        for i, (sym, res) in enumerate(zip(atom_symbol, residue_name)):
            atom_idx = self._atom_vocab_map.get(sym, len(self._atom_vocab))
            res_idx = self._residue_vocab_map.get(res, len(self._residue_vocab))
            node_feature[i, atom_idx] = 1.0
            node_feature[i, 18 + res_idx] = 1.0
        edge_list, num_relation = self._build_edges(path, atom_pos_t, atom2res_t, residue_keys, atom_name_t, atom_keys)
        protein = PygProtein(
            node_position=atom_pos_t,
            atom_name=atom_name_t,
            atom2residue=atom2res_t,
            residue_type=residue_type_t,
            residue_chain=[c for c, _ in residue_keys],
            residue_number=[n for _, n in residue_keys],
            edge_list=edge_list,
            num_relation=num_relation,
            node_feature=node_feature,
        )
        residue_identifiers = [(c, n) for c, n in residue_keys]
        if self.repack_radius is None:
            protein.repack_residue_mask = torch.ones(protein.num_residue, dtype=torch.bool, device=protein.device)
        else:
            repack_mask, _ = repack.select_residues_by_radius(
                atom_positions=protein.node_position,
                atom2residue=protein.atom2residue,
                num_residue=protein.num_residue,
                residue_identifiers=residue_identifiers,
                center_selectors=self.center_residue_selectors,
                radius=self.repack_radius,
            )
            protein.repack_residue_mask = repack_mask
        return protein

    def _build_edges(
        self,
        pdb_path: str,
        pos: torch.Tensor,
        atom2res: torch.Tensor,
        residue_keys: list[tuple[str, int]],
        atom_name: torch.Tensor,
        atom_keys: list[tuple[str, int, str]],
    ):
        bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        edges = []
        # Prefer RDKit bond parsing to match TorchDrug bond relation semantics.
        if Chem is not None:
            mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=True)
            if mol is not None:
                key_to_indices: dict[tuple[str, int, str], list[int]] = {}
                for idx, key in enumerate(atom_keys):
                    key_to_indices.setdefault(key, []).append(idx)
                rdkit_to_local = {}
                for atom in mol.GetAtoms():
                    info = atom.GetPDBResidueInfo()
                    if info is None:
                        continue
                    chain = (info.GetChainId() or "A").strip() or "A"
                    resnum = int(info.GetResidueNumber())
                    aname = (info.GetName() or "").strip()
                    key = (chain, resnum, aname)
                    candidates = key_to_indices.get(key)
                    if candidates:
                        rdkit_to_local[atom.GetIdx()] = candidates.pop(0)
                for bond in mol.GetBonds():
                    i = rdkit_to_local.get(bond.GetBeginAtomIdx())
                    j = rdkit_to_local.get(bond.GetEndAtomIdx())
                    if i is None or j is None:
                        continue
                    btype = str(bond.GetBondType())
                    rel = bond2id.get(btype, 0)
                    edges.append((i, j, rel))
                    edges.append((j, i, rel))

        # Distance fallback when bond map is incomplete / unavailable.
        if not edges:
            for rid in range(len(residue_keys)):
                idx = torch.nonzero(atom2res == rid, as_tuple=False).flatten()
                if idx.numel() < 2:
                    continue
                subpos = pos[idx]
                dist = torch.cdist(subpos, subpos)
                e = torch.nonzero((dist > 0) & (dist < 1.95), as_tuple=False)
                for a, b in e.tolist():
                    edges.append((int(idx[a]), int(idx[b]), 0))
            inv_atom_name = {v: k for k, v in rotamer.atom_name_vocab.items()}
            by_res_atom = {}
            for i in range(atom2res.numel()):
                rid = int(atom2res[i].item())
                by_res_atom.setdefault(rid, {})[inv_atom_name[int(atom_name[i].item())]] = int(i)
            for rid in range(len(residue_keys) - 1):
                c1, r1 = residue_keys[rid]
                c2, r2 = residue_keys[rid + 1]
                if c1 != c2:
                    continue
                if r2 - r1 != 1:
                    continue
                c_atom = by_res_atom.get(rid, {}).get("C")
                n_atom = by_res_atom.get(rid + 1, {}).get("N")
                if c_atom is None or n_atom is None:
                    continue
                if torch.norm(pos[c_atom] - pos[n_atom]).item() < 2.0:
                    edges.append((c_atom, n_atom, 0))
                    edges.append((n_atom, c_atom, 0))

        if not edges:
            return torch.zeros((0, 3), dtype=torch.long), 1
        edge_tensor = torch.tensor(edges, dtype=torch.long)
        num_relation = int(edge_tensor[:, 2].max().item()) + 1
        num_relation = max(num_relation, 1)
        return edge_tensor, num_relation

    def get_item(self, index):
        return {"graph": self._parse_pdb(self.pdb_files[index])}


class PygTorsionalDiffusion(nn.Module):
    NUM_CHI_ANGLES = 4
    eps = 1e-10

    def __init__(
        self,
        sigma_embedding: nn.Module,
        model: nn.Module,
        torsion_mlp_hidden_dims: list[int],
        schedule_1pi_periodic: PygSO2VESchedule,
        schedule_2pi_periodic: PygSO2VESchedule,
        graph_construction_model: Any | None = None,
        train_chi_id=None,
    ):
        super().__init__()
        import copy
        self.model_list = nn.ModuleList([copy.deepcopy(model) for _ in range(self.NUM_CHI_ANGLES)])
        self.sigma_embedding_list = nn.ModuleList([copy.deepcopy(sigma_embedding) for _ in range(self.NUM_CHI_ANGLES)])
        self.torsion_mlp_list = nn.ModuleList(
            [PygMLP(self.model_list[i].output_dim, list(torsion_mlp_hidden_dims) + [4]) for i in range(self.NUM_CHI_ANGLES)]
        )
        self.schedule_1pi_periodic = schedule_1pi_periodic
        self.schedule_2pi_periodic = schedule_2pi_periodic
        self.graph_construction_model = graph_construction_model
        self.train_chi_id = train_chi_id

    def predict(self, batch):
        protein = batch["graph"]
        chi_id = batch["chi_id"]
        sigma = batch["sigma"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(protein)
        else:
            graph = PygProteinGraph(protein=protein, edge_list=protein.edge_list, edge_feature=None, num_relation=protein.num_relation)
        node_sigma = sigma[graph.atom2graph]
        node_feature = self.sigma_embedding_list[chi_id](graph.node_feature.float(), node_sigma)
        node_feature = self.model_list[chi_id](graph, node_feature)["node_feature"]
        residue_feature = _scatter_mean(node_feature, graph.atom2residue, graph.num_residue)
        pred = self.torsion_mlp_list[chi_id](residue_feature)
        torsion_sigma = sigma[graph.residue2graph].unsqueeze(-1).expand(-1, self.NUM_CHI_ANGLES)
        score_norm_1pi = torch.tensor(self.schedule_1pi_periodic.score_norm(torsion_sigma), device=graph.device)
        score_norm_2pi = torch.tensor(self.schedule_2pi_periodic.score_norm(torsion_sigma), device=graph.device)
        score_norm = torch.where(graph.chi_1pi_periodic_mask, score_norm_1pi, score_norm_2pi)
        pred_score = pred * score_norm.sqrt()
        pred_score = pred_score * graph.chi_mask.to(pred_score.dtype)
        return pred_score, score_norm

    @torch.no_grad()
    def generate(self, batch, randomize=True):
        protein = batch["graph"]
        repack_residue_mask = getattr(protein, "repack_residue_mask", None)
        repack_chi_mask = repack_residue_mask.unsqueeze(-1).expand(-1, self.NUM_CHI_ANGLES) if repack_residue_mask is not None else None
        original = protein.clone()
        if randomize:
            protein = rotamer.randomize(protein)
            if repack_residue_mask is not None:
                keep_atom_mask = (~repack_residue_mask)[protein.atom2residue]
                protein.node_position[keep_atom_mask] = original.node_position[keep_atom_mask]
        schedule = self.schedule_1pi_periodic.reverse_t_schedule.to(protein.device)
        for chi_id in range(self.NUM_CHI_ANGLES):
            for j in range(len(schedule) - 1):
                t = schedule[j]
                dt = schedule[j] - schedule[j + 1]
                chis = rotamer.get_chis(protein, protein.node_position)
                sigma = self.schedule_1pi_periodic.t_to_sigma(t).repeat(protein.batch_size)
                chi_protein = rotamer.remove_by_chi(protein, chi_id)
                pred_score, _ = self.predict({"graph": chi_protein, "sigma": sigma, "chi_id": chi_id})
                chi_1 = chi_protein.chi_1pi_periodic_mask
                chi_2 = chi_protein.chi_2pi_periodic_mask
                if repack_chi_mask is not None:
                    chi_1 = chi_1 & repack_chi_mask
                    chi_2 = chi_2 & repack_chi_mask
                chis = self.schedule_1pi_periodic.step(chis, pred_score, t, dt, chi_1)
                chis = self.schedule_2pi_periodic.step(chis, pred_score, t, dt, chi_2)
                protein = rotamer.set_chis(protein, chis)
        return {"graph": protein}

    def get_metric(self, pred_protein, true_protein, metric):
        pred_pos = pred_protein.node_position
        true_pos = true_protein.node_position
        protein = true_protein
        pred_pos_per_residue = torch.zeros(protein.num_residue, len(rotamer.atom_name_vocab), 3, device=protein.device)
        true_pos_per_residue = torch.zeros(protein.num_residue, len(rotamer.atom_name_vocab), 3, device=protein.device)
        pred_pos_per_residue[protein.atom2residue, protein.atom_name] = pred_pos
        true_pos_per_residue[protein.atom2residue, protein.atom_name] = true_pos
        symm_true = rotamer._get_symm_atoms(true_pos_per_residue, protein.residue_type)
        rmsd = rotamer._rmsd_per_residue(pred_pos_per_residue, true_pos_per_residue, protein.sidechain37_mask)
        sym_rmsd = rotamer._rmsd_per_residue(pred_pos_per_residue, symm_true, protein.sidechain37_mask)
        replace = rmsd > sym_rmsd
        rmsd[replace] = sym_rmsd[replace]
        true_pos_per_residue[replace] = symm_true[replace]
        true_pos = true_pos_per_residue[protein.atom2residue, protein.atom_name]
        metric["atom_rmsd_per_residue"] = rmsd
        pred_chi = rotamer.get_chis(protein, pred_pos)
        true_chi = rotamer.get_chis(protein, true_pos)
        chi_diff = (pred_chi - true_chi).abs()
        chi_ae = torch.minimum(chi_diff, 2 * np.pi - chi_diff)
        chi_ae_periodic = torch.minimum(chi_ae, np.pi - chi_ae)
        chi_ae[protein.chi_1pi_periodic_mask] = chi_ae_periodic[protein.chi_1pi_periodic_mask]
        metric["chi_ae_deg"] = chi_ae[protein.chi_mask] * 180 / np.pi
        for i in range(self.NUM_CHI_ANGLES):
            metric[f"chi_{i}_ae_deg"] = chi_ae[:, i][protein.chi_mask[:, i]] * 180 / np.pi
        return metric


class PygConfidencePrediction(PygTorsionalDiffusion):
    def __init__(self, confidence_model: nn.Module, num_sample: int, num_mlp_layer: int, **kwargs):
        super().__init__(**kwargs)
        self.confidence_model = confidence_model
        self.num_sample = num_sample
        hidden = [confidence_model.output_dim] * num_mlp_layer + [1]
        self.mlp = PygMLP(confidence_model.output_dim, hidden)

    def predict_rmsd(self, batch):
        protein = batch["graph"]
        graph = self.graph_construction_model(protein) if self.graph_construction_model else PygProteinGraph(protein=protein, edge_list=protein.edge_list, edge_feature=None, num_relation=protein.num_relation)
        atom_feature = self.confidence_model(graph, graph.node_feature.float())["node_feature"]
        residue_feature = _scatter_mean(atom_feature, graph.atom2residue, graph.num_residue)
        return self.mlp(residue_feature).squeeze(-1)

    @torch.no_grad()
    def generate(self, batch, randomize=True):
        protein = batch["graph"]
        repack_residue_mask = getattr(protein, "repack_residue_mask", None)
        input_protein = protein.clone()
        best_protein = input_protein.clone()
        best_rmsd = torch.zeros(protein.num_residue, device=protein.device) + 1e6
        for _ in range(self.num_sample):
            sampled = super().generate({"graph": input_protein.clone()}, randomize=randomize)
            protein = sampled["graph"]
            rmsd = self.predict_rmsd(sampled)
            update_mask = rmsd < best_rmsd
            if repack_residue_mask is not None:
                update_mask = update_mask & repack_residue_mask
            atom_update_mask = update_mask[protein.atom2residue]
            best_protein.node_position[atom_update_mask] = protein.node_position[atom_update_mask]
            best_rmsd[update_mask] = rmsd[update_mask]
        return {"graph": best_protein, "rmsd": best_rmsd}


class PygConfigTranslator:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _expect_class(section: dict, expected: str, label: str):
        actual = section.get("class")
        if actual != expected:
            raise ValueError(f"PyG backend only supports {label} class `{expected}`, got `{actual}`")

    def build_graph_construction(self):
        section = self.cfg.task.get("graph_construction_model")
        if not section:
            return None
        self._expect_class(section, "GraphConstruction", "graph construction")
        edge_layers = []
        for edge_cfg in section.get("edge_layers", []):
            cls = edge_cfg.get("class")
            if cls == "BondEdge":
                edge_layers.append(PygBondEdge())
            elif cls == "SpatialEdge":
                edge_layers.append(
                    PygSpatialEdge(
                        radius=edge_cfg.get("radius", 5.0),
                        min_distance=edge_cfg.get("min_distance", 5),
                        max_distance=edge_cfg.get("max_distance"),
                        max_num_neighbors=edge_cfg.get("max_num_neighbors", 32),
                    )
                )
            elif cls == "KNNEdge":
                edge_layers.append(
                    PygKNNEdge(
                        k=edge_cfg.get("k", 10),
                        min_distance=edge_cfg.get("min_distance", 5),
                        max_distance=edge_cfg.get("max_distance"),
                    )
                )
            else:
                raise ValueError(f"PyG backend unsupported edge layer `{cls}`")
        return PygGraphConstruction(edge_layers=edge_layers, edge_feature=section.get("edge_feature", "gearnet"))

    def build_model(self, section_name: str):
        model_cfg = self.cfg.task.get(section_name)
        if not model_cfg:
            raise ValueError(f"Missing `{section_name}` in task config")
        self._expect_class(model_cfg, "GearNet", section_name)
        return PygGearNet(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            batch_norm=model_cfg.get("batch_norm", False),
            concat_hidden=model_cfg.get("concat_hidden", False),
            short_cut=model_cfg.get("short_cut", False),
            readout=model_cfg.get("readout", "sum"),
            num_relation=model_cfg.get("num_relation", 1),
            edge_input_dim=model_cfg.get("edge_input_dim"),
            num_angle_bin=model_cfg.get("num_angle_bin"),
        )

    def build_sigma_embedding(self):
        sigma_cfg = self.cfg.task.get("sigma_embedding")
        self._expect_class(sigma_cfg, "SigmaEmbeddingLayer", "sigma embedding")
        return PygSigmaEmbeddingLayer(
            input_dim=sigma_cfg["input_dim"],
            hidden_dims=sigma_cfg["hidden_dims"],
            sigma_dim=sigma_cfg["sigma_dim"],
            embed_type=sigma_cfg.get("embed_type", "sinusoidal"),
            operation=sigma_cfg.get("operation", "post_add"),
        )

    def build_schedule(self, key: str):
        scfg = self.cfg.task.get(key)
        self._expect_class(scfg, "SO2VESchedule", key)
        kwargs = dict(scfg)
        kwargs.pop("class", None)
        kwargs.setdefault("cache_read_only", True)
        return PygSO2VESchedule(**kwargs)

    def build_task(self):
        task_cls = self.cfg.task.get("class")
        if task_cls not in {"TorsionalDiffusion", "ConfidencePrediction"}:
            raise ValueError(f"PyG backend only supports task class TorsionalDiffusion/ConfidencePrediction, got `{task_cls}`")

        sigma_embedding = self.build_sigma_embedding()
        model = self.build_model("model")
        graph_construction_model = self.build_graph_construction()
        schedule_1pi = self.build_schedule("schedule_1pi_periodic")
        schedule_2pi = self.build_schedule("schedule_2pi_periodic")
        torsion_hidden = self.cfg.task.get("torsion_mlp_hidden_dims", [64, 128])

        if task_cls == "TorsionalDiffusion":
            return PygTorsionalDiffusion(
                sigma_embedding=sigma_embedding,
                model=model,
                torsion_mlp_hidden_dims=torsion_hidden,
                schedule_1pi_periodic=schedule_1pi,
                schedule_2pi_periodic=schedule_2pi,
                graph_construction_model=graph_construction_model,
                train_chi_id=self.cfg.task.get("train_chi_id"),
            )

        confidence_model = self.build_model("confidence_model")
        return PygConfidencePrediction(
            sigma_embedding=sigma_embedding,
            model=model,
            confidence_model=confidence_model,
            torsion_mlp_hidden_dims=torsion_hidden,
            schedule_1pi_periodic=schedule_1pi,
            schedule_2pi_periodic=schedule_2pi,
            graph_construction_model=graph_construction_model,
            num_sample=self.cfg.task.get("num_sample", 5),
            num_mlp_layer=self.cfg.task.get("num_mlp_layer", 1),
            train_chi_id=self.cfg.task.get("train_chi_id"),
        )

    def build_dataset(self):
        ds_cfg = dict(self.cfg.test_set)
        cls = ds_cfg.pop("class", None)
        if cls != "SideChainDataset":
            raise ValueError(f"PyG backend only supports dataset class `SideChainDataset`, got `{cls}`")
        transform_cfg = ds_cfg.pop("transform", None)
        if transform_cfg not in (None, {"class": "Compose", "transforms": []}):
            raise ValueError("PyG backend only supports empty transform compose in inference config")
        return PygSideChainDataset(**ds_cfg)


class NativeRunner:
    def _configure_runtime_environment(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(output_dir, ".torch_extensions"))
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".mplconfig"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(output_dir, ".cache"))
        os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _load_model_checkpoint(self, task_module: nn.Module, model_checkpoint: str):
        checkpoint = torch.load(os.path.expanduser(model_checkpoint), map_location=torch.device("cpu"))
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = task_module.load_state_dict(state, strict=False)
        if missing and len(missing) > 0:
            converted = util.convert_model_ckpt(state)
            if converted:
                task_module.load_state_dict(converted, strict=False)
        return {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }

    def run(self, request, *, backend_requested: str, backend_effective: str, backend_mode: str, fallback_reason: str | None):
        self._configure_runtime_environment(request.output_dir)
        cfg = util.load_config(os.path.realpath(request.config))
        cfg.test_set.pdb_files = request.pdb_files
        cfg.test_set.center_residues = request.center_residues
        cfg.test_set.repack_radius = request.repack_radius
        cfg.test_set.hetero_policy = request.hetero_policy
        cfg.backend = backend_effective
        cfg.cache = cfg.get("cache", {})
        cfg.cache["root"] = resolve_cache_root(request.cache_root or cfg.cache.get("root"))
        cfg.cache["mode"] = "read_only"
        if not request.cache_read_only:
            raise RuntimeError("Inference enforces read-only cache mode.")
        for skey in ("schedule_1pi_periodic", "schedule_2pi_periodic"):
            if skey in cfg.task:
                cfg.task[skey]["cache_folder"] = cfg.cache["root"]
                cfg.task[skey]["cache_read_only"] = True

        device = choose_torch_device(request.device)
        if request.fast and getattr(cfg.task, "class", "") == "ConfidencePrediction" and "num_sample" in cfg.task:
            cfg.task.num_sample = max(1, min(int(cfg.task.num_sample), 2))

        self._set_seed(request.seed)
        logger = util.get_root_logger(file=False)
        logger.warning("Backend requested: %s", backend_requested)
        logger.warning("Backend effective: %s (%s)", backend_effective, backend_mode)
        logger.warning("Device: %s", device)
        logger.warning("Config file: %s", request.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s", request.output_dir)
        logger.warning("Cache root: %s", cfg.cache["root"])
        logger.warning("Cache mode: %s", cfg.cache["mode"])
        logger.warning("Cache preflight validation: start")
        cache_validation = validate_required_schedule_caches(cfg.cache["root"])
        if cache_validation["errors"]:
            raise RuntimeError(
                "Read-only cache validation failed. "
                f"cache_root={cfg.cache['root']} errors={cache_validation['errors']}. "
                f"Run `diffpack-prepare-cache --cache_root {cfg.cache['root']}` and retry."
            )
        logger.warning("Cache preflight validation: ok")

        translator = PygConfigTranslator(cfg)
        task_module = translator.build_task()
        task_module = task_module.to(device)

        ckpt_info = None
        if "model_checkpoint" in cfg and cfg.model_checkpoint:
            ckpt_info = self._load_model_checkpoint(task_module, cfg.model_checkpoint)

        test_set = translator.build_dataset()

        profile_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) if request.profile else nullcontext()

        output_files = []
        metric_summary = {}
        start = time.perf_counter()
        with torch.no_grad(), profile_ctx as prof:
            for i in range(len(test_set)):
                item = test_set.get_item(i)
                batch = move_to_device(item, device)
                true_protein = batch["graph"].clone()
                pred = task_module.generate(batch)["graph"]
                metric = task_module.get_metric(pred, true_protein, {})
                metric_summary = {
                    "atom_rmsd_per_residue": float(metric["atom_rmsd_per_residue"].mean().item()),
                    "chi_0_mae_deg": float(metric["chi_0_ae_deg"].mean().item()),
                    "chi_1_mae_deg": float(metric["chi_1_ae_deg"].mean().item()),
                    "chi_2_mae_deg": float(metric["chi_2_ae_deg"].mean().item()),
                    "chi_3_mae_deg": float(metric["chi_3_ae_deg"].mean().item()),
                }
                pdb_file = os.path.basename(test_set.pdb_files[i])
                output_path = os.path.join(request.output_dir, pdb_file)
                pred.cpu().to_pdb(output_path)
                output_files.append(output_path)
        elapsed = time.perf_counter() - start

        profile_path = None
        if request.profile and prof is not None:
            profile_path = os.path.join(request.output_dir, "torch_profile.txt")
            with open(profile_path, "w", encoding="utf-8") as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total"))

        metadata = {
            "backend": backend_effective,
            "backend_requested": backend_requested,
            "backend_effective": backend_effective,
            "backend_mode": backend_mode,
            "fallback_reason": fallback_reason,
            "device": str(device),
            "elapsed_sec": elapsed,
            "profile_path": profile_path,
            "metrics": metric_summary,
            "output_files": output_files,
            "checkpoint_load": ckpt_info,
            "cache_root": cfg.cache["root"],
            "cache_mode": cfg.cache["mode"],
            "cache_validation_status": "pass",
            "cache_validation_errors": [],
            "cache_keys": cache_validation["keys"],
        }
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        return metadata


# Backward-compatible internal aliases for code that still references old symbol names.
PygNativeRunner = NativeRunner
NativeConfigTranslator = PygConfigTranslator
