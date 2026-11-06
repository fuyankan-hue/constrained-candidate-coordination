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


import torch


def compute_pose_dynamic_loss(traj_refine_best, gt_target, abs_vel, reg_mask,
                              loss_steps, num_historical_steps, num_agents,
                              device, output_dim, dt, pose_lr, agent_type):
    # Only apply kinematic constraint to VEHICLE agents (type == 1)
    # The bicycle model p_t = p_{t-1} + v * [cos(heading), sin(heading)] * dt
    # only holds for vehicles, not pedestrians (can move sideways/backwards)
    # or cyclists (heading not always aligned with motion direction).
    vehicle_mask = (agent_type == 0) | (agent_type == 4)

    cur_pose = torch.zeros(num_agents, 1, output_dim).to(device)
    cur_head = torch.zeros(num_agents, 1, 1).to(device)
    pred_proximal_pose = torch.concat([cur_pose, traj_refine_best[:, :loss_steps - 1, :output_dim]], dim=-2)
    gt_head = torch.concat([cur_head, gt_target[:, :loss_steps, -1].unsqueeze(-1)], dim=-2).squeeze(-1)
    abs_vel_proximal = abs_vel[:, num_historical_steps - 1:num_historical_steps + loss_steps, :].to(device)
    vel_scalar = torch.linalg.norm(abs_vel_proximal, dim=-1)

    dyn_pose = torch.zeros(pred_proximal_pose.size()).to(device)
    dyn_pose[..., 0] = pred_proximal_pose[:, :, 0] + torch.mul(vel_scalar[:, :-1], gt_head[..., :-1].cos() * dt)
    dyn_pose[..., 1] = pred_proximal_pose[:, :, 1] + torch.mul(vel_scalar[:, :-1], gt_head[..., :-1].sin() * dt)

    pose_l2_loss = (torch.norm(dyn_pose - traj_refine_best[:, :loss_steps, :output_dim], p=2, dim=-1)
                    * reg_mask[:, :loss_steps])

    # Zero out non-vehicle agents
    pose_l2_loss = pose_l2_loss * vehicle_mask.unsqueeze(-1)

    pose_lr_weight = torch.zeros(pose_l2_loss.size()).to(device)
    for i in range(pose_lr_weight.size(1)):
        pose_lr_weight[:, i] = pose_lr
    pose_l2_loss = pose_lr_weight * pose_l2_loss
    pose_l2_loss = pose_l2_loss.sum(dim=0) / vehicle_mask.sum().clamp_(min=1)
    pose_l2_loss = pose_l2_loss.mean()

    return pose_l2_loss


def compute_head_dynamic_loss(traj_refine_best, gt_target, reg_mask,
                              loss_steps, num_agents, device, output_dim,
                              output_head, head_lr, agent_type):
    # Only apply heading dynamic constraint to VEHICLE agents (type == 1).
    # The heading consistency formula θ_t = θ_{t-1} + Δθ_gt assumes
    # vehicle-like kinematics where heading changes smoothly.
    vehicle_mask = (agent_type == 0) | (agent_type == 4)

    cur_head = torch.zeros(num_agents, 1, 1).to(device)
    gt_head = torch.concat([cur_head, gt_target[:, :loss_steps, -1].unsqueeze(-1)], dim=-2).squeeze(-1)
    dt_theta = gt_head[:, :-1] - gt_head[:, 1:]

    if output_head:
        # traj_refine layout: [loc_x, loc_y, loc_head, scale_x, scale_y, conc_head]
        # Use model's direct heading prediction at index output_dim (= 2)
        pred_head = traj_refine_best[:, :loss_steps, output_dim]  # [N, T]
        # θ_t ≈ θ_{t-1} + Δθ_gt: shift pred_head by 1, prepend 0
        pred_proximal_head = torch.concat(
            [cur_head.squeeze(-1), pred_head[:, :-1]], dim=-1)  # [N, T]
        dyn_head = pred_proximal_head + dt_theta
        # Compare against model's direct heading output (loc_refine_head)
        head_target = traj_refine_best[:, :loss_steps, output_dim]  # [N, T]
    else:
        # Compute heading from predicted trajectory motion vectors (original behavior)
        cur_pose = torch.zeros(num_agents, 1, output_dim).to(device)
        whole_traj_2d = torch.cat([cur_pose, traj_refine_best[:, :loss_steps, :output_dim]], dim=-2)
        motion_vector_2d = whole_traj_2d[:, 1:, :] - whole_traj_2d[:, :-1, :]
        head_cal = torch.atan2(motion_vector_2d[..., 1], motion_vector_2d[..., 0])
        pred_proximal_head = torch.concat(
            [cur_head, head_cal[:, :-1].unsqueeze(-1)], dim=-2).squeeze(-1)  # [N, T]
        dyn_head = pred_proximal_head + dt_theta
        # original index: output_dim + output_head (= output_dim + 0 = output_dim → scale_x)
        head_target = traj_refine_best[:, :loss_steps, output_dim + output_head]  # [N, T]

    head_l2_loss = (torch.norm(dyn_head.unsqueeze(-1) - head_target.unsqueeze(-1), p=2, dim=-1)
                    * reg_mask[:, :loss_steps])

    # Zero out non-vehicle agents
    head_l2_loss = head_l2_loss * vehicle_mask.unsqueeze(-1)

    head_lr_weight = torch.zeros(head_l2_loss.size()).to(device)
    for i in range(head_lr_weight.size(1)):
        head_lr_weight[:, i] = head_lr
    head_l2_loss = head_lr_weight * head_l2_loss
    head_l2_loss = head_l2_loss.sum(dim=0) / vehicle_mask.sum().clamp_(min=1)
    head_l2_loss = head_l2_loss.mean()

    return head_l2_loss
