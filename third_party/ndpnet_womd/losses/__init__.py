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

from losses.focal_loss import FocalLoss
from losses.gaussian_nll_loss import GaussianNLLLoss
from losses.laplace_nll_loss import LaplaceNLLLoss
from losses.nll_loss import NLLLoss
from losses.von_mises_nll_loss import VonMisesNLLLoss
from losses.motion_consistency_loss import compute_pose_dynamic_loss
from losses.motion_consistency_loss import compute_head_dynamic_loss
