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
import os
import tarfile
from typing import Any, List, Mapping

import numpy as np

try:
    from waymo_open_dataset.protos import motion_submission_pb2
    from waymo_open_dataset.protos import sim_agents_submission_pb2
except ImportError:
    motion_submission_pb2 = object
    sim_agents_submission_pb2 = object


def generate_waymo_motion_prediction_submission(
        predictions: Mapping[str, Any],
        account_name: str,
        method_name: str,
        authors: List[str],
        affiliation: str,
        description: str,
        method_link: str,
        uses_lidar_data: bool,
        uses_camera_data: bool,
        uses_public_model_pretraining: bool,
        public_model_names: List[str],
        num_model_parameters: str,
        path: str,
        joint: bool = False) -> None:
    submission = motion_submission_pb2.MotionChallengeSubmission()
    if joint:
        submission.submission_type = motion_submission_pb2.MotionChallengeSubmission.INTERACTION_PREDICTION
    else:
        submission.submission_type = motion_submission_pb2.MotionChallengeSubmission.MOTION_PREDICTION
    submission.account_name = account_name
    submission.unique_method_name = method_name
    submission.authors.extend(authors)
    submission.affiliation = affiliation
    submission.description = description
    submission.method_link = method_link
    submission.uses_lidar_data = uses_lidar_data
    submission.uses_camera_data = uses_camera_data
    submission.uses_public_model_pretraining = uses_public_model_pretraining
    submission.public_model_names.extend(public_model_names)
    submission.num_model_parameters = num_model_parameters
    for scenario_id, prediction in predictions.items():
        scenario_prediction = motion_submission_pb2.ChallengeScenarioPredictions()
        scenario_prediction.scenario_id = scenario_id
        if joint:
            joint_trajectories, confidences = prediction
            num_modes = len(confidences)
            for k in range(num_modes):
                scored_joint_trajectory = motion_submission_pb2.ScoredJointTrajectory()
                scored_joint_trajectory.confidence = confidences[k].item()
                for track_id, joint_trajectory in joint_trajectories.items():
                    object_trajectory = motion_submission_pb2.ObjectTrajectory()
                    object_trajectory.object_id = track_id
                    object_trajectory.trajectory.center_x.extend(joint_trajectory[k, :, 0].tolist())
                    object_trajectory.trajectory.center_y.extend(joint_trajectory[k, :, 1].tolist())
                    scored_joint_trajectory.trajectories.append(object_trajectory)
                scenario_prediction.joint_prediction.joint_trajectories.append(scored_joint_trajectory)
        else:
            for track_id, (trajectories, confidences) in prediction.items():
                single_object_prediction = motion_submission_pb2.SingleObjectPrediction()
                single_object_prediction.object_id = track_id
                for trajectory, confidence in zip(trajectories, confidences):
                    scored_trajectory = motion_submission_pb2.ScoredTrajectory()
                    scored_trajectory.confidence = confidence.item()
                    scored_trajectory.trajectory.center_x.extend(trajectory[:, 0].tolist())
                    scored_trajectory.trajectory.center_y.extend(trajectory[:, 1].tolist())
                    single_object_prediction.trajectories.append(scored_trajectory)
                scenario_prediction.single_predictions.predictions.append(single_object_prediction)
        submission.scenario_predictions.append(scenario_prediction)
    with open(path, 'wb') as file:
        file.write(submission.SerializeToString())
    with tarfile.open(f'{os.path.splitext(path)[0]}.tar.gz', 'w:gz') as tar:
        tar.add(path, arcname=os.path.basename(path))


def generate_waymo_simulation_submission(
        predictions: Mapping[str, Any],
        num_rollouts: int,
        account_name: str,
        method_name: str,
        authors: List[str],
        affiliation: str,
        description: str,
        method_link: str,
        uses_lidar_data: bool,
        uses_camera_data: bool,
        uses_public_model_pretraining: bool,
        public_model_names: List[str],
        num_model_parameters: str,
        acknowledge_complies_with_closed_loop_requirement: bool,
        submission_dir: str,
        submission_file_name: str) -> None:
    scenario_rollouts = []
    for scenario_id, prediction in predictions.items():
        scenario_rollout = sim_agents_submission_pb2.ScenarioRollouts()
        scenario_rollout.scenario_id = scenario_id
        joint_scenes = [sim_agents_submission_pb2.JointScene() for _ in range(num_rollouts)]
        for track_id, trajectories in prediction.items():
            for rollout_idx, trajectory in enumerate(trajectories):
                simulated_trajectory = sim_agents_submission_pb2.SimulatedTrajectory()
                simulated_trajectory.object_id = track_id
                simulated_trajectory.center_x.extend(trajectory[:, 0].tolist())
                simulated_trajectory.center_y.extend(trajectory[:, 1].tolist())
                simulated_trajectory.center_z.extend(trajectory[:, 2].tolist())
                simulated_trajectory.heading.extend(trajectory[:, 3].tolist())
                joint_scenes[rollout_idx].simulated_trajectories.append(simulated_trajectory)
        scenario_rollout.joint_scenes.extend(joint_scenes)
        scenario_rollouts.append(scenario_rollout)
    ext_submission_file_names = []
    for shard_num, shard_scenario_rollouts in enumerate(np.array_split(scenario_rollouts, 150)):
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission()
        shard_submission.scenario_rollouts.extend(shard_scenario_rollouts.tolist())
        shard_submission.submission_type = sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION
        shard_submission.account_name = account_name
        shard_submission.unique_method_name = method_name
        shard_submission.authors.extend(authors)
        shard_submission.affiliation = affiliation
        shard_submission.description = description
        shard_submission.method_link = method_link
        shard_submission.uses_lidar_data = uses_lidar_data
        shard_submission.uses_camera_data = uses_camera_data
        shard_submission.uses_public_model_pretraining = uses_public_model_pretraining
        shard_submission.public_model_names.extend(public_model_names)
        shard_submission.num_model_parameters = num_model_parameters
        shard_submission.acknowledge_complies_with_closed_loop_requirement = acknowledge_complies_with_closed_loop_requirement
        shard_num_str = f'00000{shard_num + 1}'
        ext_submission_file_name = f'{submission_file_name}.binproto-{shard_num_str[-5:]}-of-00150'
        with open(os.path.join(submission_dir, ext_submission_file_name), 'wb') as file:
            file.write(shard_submission.SerializeToString())
        ext_submission_file_names.append(ext_submission_file_name)
    with tarfile.open(os.path.join(submission_dir, f'{submission_file_name}.tar.gz'), 'w:gz') as tar:
        for ext_submission_file_name in ext_submission_file_names:
            tar.add(os.path.join(submission_dir, ext_submission_file_name), arcname=ext_submission_file_name)
