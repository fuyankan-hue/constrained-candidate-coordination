# Copyright (c) 2026, Hua Hu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code base from QCNet (CVPR 2023, https://github.com/ZikangZhou/QCNet)

import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle


class NDPNetDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps_lat: int,
                 num_recurrent_steps_prox: int,
                 num_proximal_steps:int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(NDPNetDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        # proximal + latter time steps = num future steps
        self.num_proximal_steps = num_proximal_steps
        self.num_modes = num_modes
        self.num_recurrent_steps_lat = num_recurrent_steps_lat
        self.num_recurrent_steps_prox = num_recurrent_steps_prox
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.mode_emb2 = nn.Embedding(num_modes, hidden_dim)

        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers_proximal = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers_proximal = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers_proximal = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer_proximal = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc_propose_pos_proximal = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.num_proximal_steps * output_dim // num_recurrent_steps_prox )
        self.to_scale_propose_pos_proximal = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=self.num_proximal_steps * output_dim // num_recurrent_steps_prox)
        
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head_proximal = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim= self.num_proximal_steps // num_recurrent_steps_prox )
            self.to_conc_propose_head_proximal = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim= self.num_proximal_steps // num_recurrent_steps_prox )
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head_proximal = None
            self.to_conc_propose_head_proximal = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
 
        self.num_lat_steps = num_future_steps - self.num_proximal_steps
        self.y_emb_lat = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb_hlat = nn.Parameter(torch.zeros(1, hidden_dim))
        self.traj_emb_lat = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        
        self.t2m_propose_attn_layers_lat = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers_lat = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers_lat = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer_lat = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        
        self.prox2lat_propose_atten_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=True, has_pos_emb=False)

        self.to_loc_propose_pos_lat = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim= self.num_lat_steps * output_dim // num_recurrent_steps_lat)
        self.to_scale_propose_pos_lat = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim= self.num_lat_steps * output_dim // num_recurrent_steps_lat)
        if output_head:
            self.to_loc_propose_head_lat = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim= self.num_lat_steps // num_recurrent_steps_lat)
            self.to_conc_propose_head_lat = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim= self.num_lat_steps // num_recurrent_steps_lat)
        else:
            self.to_loc_propose_head_lat = None
            self.to_conc_propose_head_lat = None

        self.m_embedding = MLPLayer(input_dim=self.num_proximal_steps * self.hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # current pos 
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)
        # 
        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)

        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1)
    
        # get mask 
        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = mask_src.any(dim=-1, keepdim=True).repeat(1, self.num_modes)
        # get r_t2m time to mode src relavant embedding
        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)

        # bipartite_dense_to_sparse 
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))

        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)

        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]

        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)

        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

        locs_propose_pos_proximal: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps_prox
        scales_propose_pos_proximal: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps_prox
        locs_propose_head_proximal: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps_prox
        concs_propose_head_proximal: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps_prox
        
        for t in range(self.num_recurrent_steps_prox):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m_propose_attn_layers_proximal[i]((x_t, m), r_t2m, edge_index_t2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_propose_attn_layers_proximal[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_propose_attn_layers_proximal[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.m2m_propose_attn_layer_proximal(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos_proximal[t] = self.to_loc_propose_pos_proximal(m)
            scales_propose_pos_proximal[t] = self.to_scale_propose_pos_proximal(m)
            if self.output_head:
                locs_propose_head_proximal[t] = self.to_loc_propose_head_proximal(m)
                concs_propose_head_proximal[t] = self.to_conc_propose_head_proximal(m)

        loc_propose_pos_proximal = torch.cumsum(torch.cat(locs_propose_pos_proximal,dim=-1).view(-1, self.num_modes, self.num_proximal_steps, self.output_dim)
                                                , dim=-2)
        scale_propose_pos_proximal = torch.cumsum(
            F.elu_(torch.cat(scales_propose_pos_proximal,dim=-1).view(-1, self.num_modes, self.num_proximal_steps, self.output_dim),
                alpha=1.0) 
                + 1.0, 
                dim=-2) + 0.1      
        if self.output_head:
            loc_propose_head_proximal = torch.cumsum(torch.tanh(torch.cat(locs_propose_head_proximal,dim=-1).unsqueeze(-1)) * math.pi, dim=-2)

            conc_propose_head_proximal = 1.0 / (torch.cumsum((F.elu_(torch.cat(concs_propose_head_proximal,dim=-1).unsqueeze(-1)) + 1.0), dim=-2) + 0.02)
            # shape: [N_A, m, T_prox, 3] => [N_A * m * T_prox, 3] => [N_A * m * T_prox, h]

            m = self.y_emb(torch.cat([loc_propose_pos_proximal.detach(),
                                      wrap_angle(loc_propose_head_proximal.detach())], dim=-1).view(-1, self.output_dim + 1))
            
            Prox_end_point  = torch.cat([loc_propose_pos_proximal[:, :, -1, :].detach(),
                                      wrap_angle(loc_propose_head_proximal[:, :, -1, :].detach())], dim=-1).view(-1, self.output_dim + 1)


        else:
            loc_propose_head_proximal = loc_propose_pos_proximal.new_zeros((loc_propose_pos_proximal.size(0), self.num_modes,
                                                          self.num_proximal_steps, 1))
            conc_propose_head_proximal = scale_propose_pos_proximal.new_zeros((scale_propose_pos_proximal.size(0), self.num_modes,
                                                             self.num_proximal_steps, 1))
            m = self.y_emb(loc_propose_pos_proximal.detach().view(-1, self.output_dim))

        m = m.reshape(-1, self.num_proximal_steps, self.hidden_dim)
        m_emb = self.m_embedding(m.reshape(-1,self.num_proximal_steps*self.hidden_dim))
        m = m.transpose(0,1)
        m2 = self.mode_emb2.weight.repeat(scene_enc['x_a'].size(0), 1)

        locs_propose_pos_lat: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps_lat
        scales_propose_pos_lat: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps_lat
        locs_propose_head_lat: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps_lat
        concs_propose_head_lat: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps_lat

        m2 = self.prox2lat_propose_atten_layer((m_emb, m2), None, edge_index_m2m)
        for t in range(self.num_recurrent_steps_lat):
            for i in range(self.num_layers):
                m2 = m2.reshape(-1, self.hidden_dim)
                m2 = self.t2m_propose_attn_layers_lat[i]((x_t, m2), r_t2m, edge_index_t2m)
                m2 = m2.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m2 = self.pl2m_propose_attn_layers_lat[i]((x_pl, m2), r_pl2m, edge_index_pl2m)
                m2 = self.a2m_propose_attn_layers_lat[i]((x_a, m2), r_a2m, edge_index_a2m)
                m2 = m2.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m2 = self.m2m_propose_attn_layer_lat(m2, None, edge_index_m2m)
            m2 = m2.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos_lat[t] = self.to_loc_propose_pos_lat(m2)
            scales_propose_pos_lat[t] = self.to_scale_propose_pos_lat(m2)
            if self.output_head:
                locs_propose_head_lat[t] = self.to_loc_propose_head_lat(m2)
                concs_propose_head_lat[t] = self.to_conc_propose_head_lat(m2)

        loc_propose_pos_lat = torch.cumsum(
            torch.cat(locs_propose_pos_lat, dim=-1).view(-1, self.num_modes, self.num_lat_steps, self.output_dim),
            dim=-2) + loc_propose_pos_proximal[:, :, -1, :].unsqueeze(2).detach()
        scale_propose_pos_lat = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos_lat, dim=-1).view(-1, self.num_modes, self.num_lat_steps, self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1        
        if self.output_head:
            loc_propose_head_lat = torch.cumsum(torch.tanh(torch.cat(locs_propose_head_lat, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2) + loc_propose_head_proximal[:, :, -1, :].unsqueeze(-2).detach()
            conc_propose_head_lat = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head_lat, dim=-1).unsqueeze(-1)) + 1.0,
                                                    dim=-2) + 0.02)
            m2 = self.y_emb_lat(torch.cat([loc_propose_pos_lat.detach(),
                                      wrap_angle(loc_propose_head_lat.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head_lat = loc_propose_pos_lat.new_zeros((loc_propose_pos_lat.size(0), self.num_modes,
                                                          self.num_lat_steps, 1))
            conc_propose_head_lat = scale_propose_pos_lat.new_zeros((scale_propose_pos_lat.size(0), self.num_modes,
                                                             self.num_lat_steps, 1))
            m2 = self.y_emb_lat(loc_propose_pos_lat.detach().view(-1, self.output_dim))
        loc_propose_pos_all = torch.concat([loc_propose_pos_proximal, loc_propose_pos_lat], dim=-2)
        scale_propose_pos_all = torch.concat([scale_propose_pos_proximal, scale_propose_pos_lat], dim=-2)
        loc_propose_head_all = torch.concat([loc_propose_head_proximal, loc_propose_head_lat], dim=-2)
        conc_propose_head_all = torch.concat([conc_propose_head_proximal, conc_propose_head_lat], dim=-2)

        m2 = m2.reshape(-1, self.num_lat_steps, self.hidden_dim).transpose(0, 1)
        m_ref = torch.concat([m,m2],dim=0)
        m_ref = self.traj_emb_lat(m_ref, self.traj_emb_hlat.unsqueeze(1).repeat(1, m_ref.size(1), 1))[1].squeeze(0)

        ## trajecoty refinement
        for i in range(self.num_layers):
            m_ref = self.t2m_refine_attn_layers[i]((x_t, m_ref), r_t2m, edge_index_t2m)
            m_ref = m_ref.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m_ref = self.pl2m_refine_attn_layers[i]((x_pl, m_ref), r_pl2m, edge_index_pl2m)
            m_ref = self.a2m_refine_attn_layers[i]((x_a, m_ref), r_a2m, edge_index_a2m)
            m_ref = m_ref.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        m_ref = self.m2m_refine_attn_layer(m_ref, None, edge_index_m2m)
        m_ref = m_ref.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m_ref).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + torch.concat([loc_propose_pos_proximal.detach(),loc_propose_pos_lat.detach()],dim=2)
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m_ref).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m_ref).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + torch.concat([loc_propose_head_proximal.detach(), loc_propose_head_lat.detach()], dim=-2)
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m_ref).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.num_future_steps, 1))
        pi = self.to_pi(m_ref).squeeze(-1)

        return {
            'loc_propose_pos_all': loc_propose_pos_all,
            'scale_propose_pos_all': scale_propose_pos_all,
            'loc_propose_head_all': loc_propose_head_all,
            'conc_propose_head_all': conc_propose_head_all,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'pi': pi,
        }
