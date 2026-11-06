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
from typing import Optional, Tuple

import torch


def nms(
        traj: torch.Tensor,
        pi: torch.Tensor,
        nms_thres: Optional[float] = None,
        vel: Optional[torch.Tensor] = None,
        max_guesses: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    default_nms_thres = traj.new_tensor([[2.0, 1.0], [3.6, 1.8], [6.0, 3.0]])
    if traj.dim() == 3:
        traj = traj.unsqueeze(0)
    num_steps = traj.size(-2)
    eval_inds = traj.new_tensor([num_steps * 3 // 8 - 1, num_steps * 5 // 8 - 1, num_steps - 1], dtype=torch.long)
    order = torch.sort(pi, dim=-1, descending=True)[1]
    keep = []
    while len(order) > 0 and len(keep) < max_guesses:
        keep.append(order[0].item())
        if nms_thres is not None:
            is_overlap = (torch.norm(traj[:, order[0], -1, :2].unsqueeze(-2) -
                                     traj[:, order[1:], -1, :2], p=2, dim=-1) < nms_thres)
        else:
            motion_vector = traj[:, order[0], eval_inds] - traj[:, order[0], eval_inds - 1]
            head = torch.atan2(motion_vector[..., 1], motion_vector[..., 0])
            cos, sin = head.cos(), head.sin()
            rot_mat = torch.stack([torch.stack([cos, -sin], dim=-1),
                                   torch.stack([sin, cos], dim=-1)],
                                  dim=-2)
            displacement_vector = torch.matmul((traj[:, order[0], eval_inds, :2].unsqueeze(1) -
                                                traj[:, order[1:]][..., eval_inds, :2]).transpose(1, 2),
                                               rot_mat)
            if vel.dim() == 1:
                vel = vel.unsqueeze(0)
            vel_magnitude = torch.norm(vel[:, :2], p=2, dim=-1)
            scale = (vel_magnitude - 1.4) * 0.5 / (11.0 - 1.4) + 0.5
            scale[vel_magnitude < 1.4] = 0.5
            scale[vel_magnitude > 11.0] = 1.0
            is_overlap = (torch.abs(displacement_vector) <
                          default_nms_thres.reshape(1, 3, 1, 2) * scale.reshape(-1, 1, 1, 1)).all(dim=-1).any(dim=-2)
        is_overlap = is_overlap.all(dim=0)
        pi[order[0]] += pi[order[torch.where(is_overlap)[0] + 1]].sum()
        pi[order[torch.where(is_overlap)[0] + 1]] *= 1e-5
        order = order[torch.where(~is_overlap)[0] + 1]
    pi /= pi.sum()
    return traj[:, keep].squeeze(0), pi[keep]
