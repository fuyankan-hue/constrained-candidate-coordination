import torch
from torch_geometric.transforms import BaseTransform

class AV2QCNetAdapter(BaseTransform):
    def __init__(self, historical_steps=50, future_steps=60):
        self.h = historical_steps
        self.f = future_steps

    def __call__(self, data):
        agent = data["agent"]
        types = agent["type"].long()
        category = agent["category"].long()

        if agent["position"].shape[1] != self.h + self.f:
            raise ValueError("Unexpected AV2 time length")

        if types.numel() == 0 or ((types < 0) | (types >= 10)).any():
            raise ValueError("Invalid AV2 agent types")

        focal = category == 3

        if int(focal.sum()) != 1:
            raise ValueError("Expected exactly one focal agent")

        if "target_mask" in agent or "interact_mask" in agent:
            raise ValueError("AV2 adaptation already applied")

        # Retain AV2-native type IDs for QCNet's 10-class encoder.
        # Convert single-scene AV2 index to a 1D tensor.
        av_index = torch.as_tensor(
            agent["av_index"],
            dtype=torch.long,
            device=types.device,
        ).reshape(-1)

        if av_index.numel() != 1:
            raise ValueError("Expected one AV index per scene")

        if not (0 <= int(av_index[0]) < int(agent["num_nodes"])):
            raise ValueError("Invalid AV index")

        agent["av_index"] = av_index

        agent["target_mask"] = focal
        agent["interact_mask"] = torch.zeros_like(focal)

        return data
