import torch
from torch import nn


class NearFarInteraction(nn.Module):
    """
    输入:
      near, far, refine: [N,6,H]
      world: [N,6,80,2]
      edges: [E,2]
    输出:
      compatibility: [E,6,6]

    无GT输入，具有候选对交换对称性。
    """

    def __init__(self, hidden_dim=128, dim=32):
        super().__init__()

        self.near_proj = nn.Linear(hidden_dim, dim)
        self.far_proj = nn.Linear(hidden_dim, dim)
        self.ref_proj = nn.Linear(hidden_dim, dim)

        pair_dim = dim * 3 + 4

        self.near_net = nn.Sequential(
            nn.Linear(pair_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        self.far_net = nn.Sequential(
            nn.Linear(pair_dim + 32, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        self.ref_net = nn.Sequential(
            nn.Linear(pair_dim + 32, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        self.head = nn.Linear(32, 1)

    @staticmethod
    def pair_embedding(features, edges, projection, geometry):
        a = projection(features[edges[:, 0]]).unsqueeze(2)
        b = projection(features[edges[:, 1]]).unsqueeze(1)

        return torch.cat([
            a + b,
            torch.abs(a - b),
            a * b,
            geometry,
        ], dim=-1)

    def forward(self, near, far, refine, world, edges):
        assert near.ndim == 3
        assert near.shape == far.shape == refine.shape
        assert near.shape[1] == 6
        assert world.shape == (near.shape[0], 6, 80, 2)
        assert edges.ndim == 2 and edges.shape[1] == 2

        if edges.shape[0] == 0:
            return world.new_zeros((0, 6, 6))

        i, j = edges[:, 0], edges[:, 1]

        distance = torch.linalg.vector_norm(
            world[i].unsqueeze(2) -
            world[j].unsqueeze(1),
            dim=-1
        )

        dn = distance[..., :30]
        df = distance[..., 30:]

        near_geometry = torch.stack([
            dn.amin(dim=-1) / 20,
            dn[..., -1] / 20,
            (dn[..., 0] - dn[..., -1]) / 20,
            dn[..., 10] / 20,
        ], dim=-1)

        far_geometry = torch.stack([
            df.amin(dim=-1) / 20,
            df[..., -1] / 20,
            (df[..., 0] - df[..., -1]) / 20,
            df[..., 19] / 20,
        ], dim=-1)

        full_geometry = torch.stack([
            distance.amin(dim=-1) / 20,
            distance[..., 19] / 20,
            distance[..., 59] / 20,
            distance[..., 79] / 20,
        ], dim=-1)

        z_near = self.near_net(
            self.pair_embedding(
                near, edges,
                self.near_proj,
                near_geometry
            )
        )

        z_far = self.far_net(
            torch.cat([
                self.pair_embedding(
                    far, edges,
                    self.far_proj,
                    far_geometry
                ),
                z_near,
            ], dim=-1)
        )

        z_ref = self.ref_net(
            torch.cat([
                self.pair_embedding(
                    refine, edges,
                    self.ref_proj,
                    full_geometry
                ),
                z_far,
            ], dim=-1)
        )

        score = self.head(z_ref).squeeze(-1)

        return score
