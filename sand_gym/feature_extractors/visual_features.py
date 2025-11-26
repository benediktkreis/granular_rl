import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GatedCNNFeatureExtractorWithToolMask(BaseFeaturesExtractor):
    """
    A feature extractor that:
      1) Concatenates heightmap_diff and tool_mask along the channel dim.
      2) Processes that with a small CNN + FC.
      3) Processes goal_mask with its own CNN + FC.
      4) Gates the diff‐path by the mask‐path: diff_feats * sigmoid(mask_feats).
      5) Runs any remaining obs through an MLP.
      6) Concats everything and projects to features_dim.
    Works whether or not you’ve applied VecFrameStack (which widens each H×W map).
    """

    def __init__(self, observation_space: gym.spaces.Dict, config: dict):
        features_dim = int(config["features_dim"])
        super().__init__(observation_space, features_dim)

        # Which keys go to the CNNs vs the MLP
        if "reconstructed_heightmap_diff" in observation_space.spaces:
            self.exclude_keys = ["reconstructed_heightmap_diff", "tool_mask", "goal_mask"]
        else:
            self.exclude_keys = ["heightmap_diff", "tool_mask", "goal_mask"]
        self.other_keys   = [k for k in observation_space.spaces if k not in self.exclude_keys]

        # — Detect the current spatial shapes —
        # heightmap_diff and tool_mask should share height, goal_mask may also be widened
        if "reconstructed_heightmap_diff" in observation_space.spaces:
            diff_h, diff_w = observation_space.spaces["reconstructed_heightmap_diff"].shape
        else:
            diff_h, diff_w = observation_space.spaces["heightmap_diff"].shape
        tool_h, tool_w = observation_space.spaces["tool_mask"].shape
        mask_h, mask_w = observation_space.spaces["goal_mask"].shape

        assert diff_h == tool_h, "heightmap_diff and tool_mask must have same height"
        assert diff_h == mask_h, "heightmap_diff and goal_mask must have same height"

        # After two stride‐2 convs: H→H/4, W→W/4
        flat_h_diff = diff_h // 4
        flat_w_diff = diff_w // 4
        flat_size_diff = 16 * flat_h_diff * flat_w_diff

        flat_h_mask = mask_h // 4
        flat_w_mask = mask_w // 4
        flat_size_mask = 16 * flat_h_mask * flat_w_mask

        # — CNN + FC for diff+tool path —
        # always 2 input channels (heightmap_diff + tool_mask)
        self.cnn_diff = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8,  kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_diff = nn.Sequential(
            nn.Linear(flat_size_diff, 64),
            nn.ReLU()
        )

        # — CNN + FC for goal_mask (gating) —
        # always 1 input channel
        self.cnn_mask = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,  kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mask = nn.Sequential(
            nn.Linear(flat_size_mask, 64),
            nn.ReLU()
        )

        # — MLP for any remaining (non-image) observations —
        self.other_obs_total_dim = 0
        for k in self.other_keys:
            shape_k = observation_space.spaces[k].shape
            self.other_obs_total_dim += int(torch.prod(torch.tensor(shape_k)))

        if self.other_obs_total_dim > 0:
            self.mlp_other = nn.Sequential(
                nn.Linear(self.other_obs_total_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        else:
            self.mlp_other = None

        # — Final projection —
        proj_in = 64 + (64 if self.mlp_other is not None else 0)
        self.projection = nn.Sequential(
            nn.Linear(proj_in, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        if "reconstructed_heightmap_diff" in observations:
            B = observations["reconstructed_heightmap_diff"].shape[0]
        else:
            B = observations["heightmap_diff"].shape[0]

        # A) Build diff+tool tensor
        if "reconstructed_heightmap_diff" in observations:
            hm = observations["reconstructed_heightmap_diff"]
        else:
            hm = observations["heightmap_diff"]
        tm = observations["tool_mask"]
        # if [B,H,W] → make [B,1,H,W]
        if hm.ndim == 3:
            hm = hm.unsqueeze(1)
        if tm.ndim == 3:
            tm = tm.unsqueeze(1)
        x = torch.cat([hm, tm], dim=1)      # [B,2,H,W_total]
        x = self.cnn_diff(x)                # [B,16,H/4,W_total/4]
        x = x.view(B, -1)                   # [B, flat_size_diff]
        diff_feats = self.fc_diff(x)        # [B,64]

        # B) Build mask tensor
        gm = observations["goal_mask"]
        if gm.ndim == 3:
            gm = gm.unsqueeze(1)
        m = self.cnn_mask(gm)
        m = m.view(B, -1)                   # [B, flat_size_mask]
        mask_feats = self.fc_mask(m)        # [B,64]

        # Gate
        gated_feats = diff_feats * torch.sigmoid(mask_feats)  # [B,64]

        # C) Other obs MLP
        if self.mlp_other:
            others = []
            for k in self.other_keys:
                t = observations[k].view(B, -1)
                others.append(t)
            other_cat = torch.cat(others, dim=1)
            other_feats = self.mlp_other(other_cat)          # [B,64]
        else:
            other_feats = torch.zeros(B, 0, device=gated_feats.device)

        # D) Final concat & project
        out = torch.cat([gated_feats, other_feats], dim=1)    # [B, 64+64]
        return self.projection(out)                           # [B, features_dim]

class CNNFeatureExtractorWithMasks(BaseFeaturesExtractor):
    """
    A feature extractor that:
      1) Concatenates heightmap_diff (or reconstructed_heightmap_diff), tool_mask, and goal_mask along the channel dim.
      2) Processes that with a small CNN + FC.
      3) Runs any remaining obs through an MLP.
      4) Concats everything and projects to features_dim.
    Works whether or not you've applied VecFrameStack.
    """

    def __init__(self, observation_space: gym.spaces.Dict, config: dict):
        features_dim = int(config["features_dim"])
        super().__init__(observation_space, features_dim)

        # Identify image keys
        if "reconstructed_heightmap_diff" in observation_space.spaces:
            diff_key = "reconstructed_heightmap_diff"
        else:
            diff_key = "heightmap_diff"
        self.diff_key = diff_key
        self.tool_key = "tool_mask"
        self.goal_key = "goal_mask"
        self.other_keys = [k for k in observation_space.spaces if k not in [diff_key, self.tool_key, self.goal_key]]

        # Spatial shape (assumed equal for all three)
        h, w = observation_space.spaces[diff_key].shape
        # After two stride-2 convs: H→H/4, W→W/4
        flat_h = h // 4
        flat_w = w // 4
        flat_size = 16 * flat_h * flat_w

        # --- CNN + FC for diff+tool+goal path ---
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU()
        )

        # --- MLP for any remaining (non-image) observations ---
        self.other_obs_total_dim = 0
        for k in self.other_keys:
            shape_k = observation_space.spaces[k].shape
            self.other_obs_total_dim += int(torch.prod(torch.tensor(shape_k)))

        if self.other_obs_total_dim > 0:
            self.mlp_other = nn.Sequential(
                nn.Linear(self.other_obs_total_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        else:
            self.mlp_other = None

        # --- Final projection ---
        proj_in = 64 + (64 if self.mlp_other is not None else 0)
        self.projection = nn.Sequential(
            nn.Linear(proj_in, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # Batch size
        B = observations[self.diff_key].shape[0]

        # Prepare heightmap_diff / reconstructed_heightmap_diff
        hm = observations[self.diff_key]
        if hm.ndim == 3:
            hm = hm.unsqueeze(1)
        # Prepare tool_mask
        tm = observations[self.tool_key]
        if tm.ndim == 3:
            tm = tm.unsqueeze(1)
        # Prepare goal_mask
        gm = observations[self.goal_key]
        if gm.ndim == 3:
            gm = gm.unsqueeze(1)

        # Concatenate into a 3-channel tensor
        x = torch.cat([hm, tm, gm], dim=1)  # [B,3,H,W]
        x = self.cnn(x)                      # [B,16,H/4,W/4]
        x = x.view(B, -1)                   # [B, flat_size]
        img_feats = self.fc(x)              # [B,64]

        # Process other obs
        if self.mlp_other:
            others = [observations[k].view(B, -1) for k in self.other_keys]
            other_cat = torch.cat(others, dim=1)
            other_feats = self.mlp_other(other_cat)  # [B,64]
        else:
            other_feats = torch.zeros(B, 0, device=img_feats.device)

        # Final concat & project
        out = torch.cat([img_feats, other_feats], dim=1)  # [B,64+64]
        return self.projection(out)  # [B, features_dim]

class CNNFeatureExtractorWithMasksAndImage(BaseFeaturesExtractor):
    """
    A feature extractor that:
      1) Uses a single depth image (key containing "depth" or "image").
      2) Processes depth through a C-channel CNN + FC at its native resolution.
      3) Processes tool_mask + goal_mask (2 channels) through a separate CNN + FC at their native resolution.
      4) Runs any remaining non-image obs through an MLP.
      5) Concatenates img_feats, mask_feats, and other_feats, then projects to features_dim.
    Masks and depth can have different spatial resolutions.
    Depth is expected as shape (H, W, C) or (H, W) in the observation space.
    """

    def __init__(self, observation_space: gym.spaces.Dict, config: dict):
        features_dim = int(config["features_dim"])
        super().__init__(observation_space, features_dim)

        # Identify depth/image and mask keys
        depth_keys = [k for k in observation_space.spaces if ('depth' in k.lower() or 'image' in k.lower())]
        if not depth_keys:
            raise ValueError("No depth or image key found in observation_space")
        self.img_key = depth_keys[0]
        self.tool_key = "tool_mask"
        self.goal_key = "goal_mask"

        # Other non-image keys
        self.other_keys = [k for k in observation_space.spaces
                           if k not in [self.img_key, self.tool_key, self.goal_key]]

        import math
        # Depth image resolution and channels
        shape_img = observation_space.spaces[self.img_key].shape
        if len(shape_img) == 3:
            h_img, w_img, c_img = shape_img
        elif len(shape_img) == 2:
            h_img, w_img = shape_img
            c_img = 1
        else:
            raise ValueError(f"Unsupported depth shape {shape_img}")
        self.img_channels = c_img

        # Helper to compute conv output dims
        def conv_out_dim(size, kernel=3, stride=2, padding=1):
            return math.floor((size + 2*padding - (kernel - 1) - 1) / stride + 1)

        # Calculate flattened size after 2 convs for depth
        h1 = conv_out_dim(h_img)
        h2 = conv_out_dim(h1)
        w1 = conv_out_dim(w_img)
        w2 = conv_out_dim(w1)
        img_flat_size = 16 * h2 * w2

        # Mask resolution (assumed same for tool and goal)
        h_mask, w_mask = observation_space.spaces[self.tool_key].shape
        # Calculate flattened size after 2 convs for masks
        hm1 = conv_out_dim(h_mask)
        hm2 = conv_out_dim(hm1)
        wm1 = conv_out_dim(w_mask)
        wm2 = conv_out_dim(wm1)
        mask_flat_size = 16 * hm2 * wm2

        # --- CNN + FC for depth image (C channels) ---
        self.cnn_img = nn.Sequential(
            nn.Conv2d(self.img_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_img = nn.Sequential(
            nn.Linear(img_flat_size, 64),
            nn.ReLU()
        )

        # --- CNN + FC for tool+goal masks (2 channels) ---
        self.cnn_mask = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mask = nn.Sequential(
            nn.Linear(mask_flat_size, 64),
            nn.ReLU()
        )

        # --- MLP for other observations ---
        self.other_obs_total_dim = 0
        for k in self.other_keys:
            shape_k = observation_space.spaces[k].shape
            self.other_obs_total_dim += int(torch.prod(torch.tensor(shape_k)))
        if self.other_obs_total_dim > 0:
            self.mlp_other = nn.Sequential(
                nn.Linear(self.other_obs_total_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
        else:
            self.mlp_other = None

        # --- Final projection ---
        proj_in = 64 + 64 + (64 if self.mlp_other is not None else 0)
        self.projection = nn.Sequential(
            nn.Linear(proj_in, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        B = observations[self.img_key].shape[0]

        # Depth image: handle channels-last or channels-first
        img = observations[self.img_key]
        if img.ndim == 4:
            if img.shape[-1] == self.img_channels:
                img = img.permute(0, 3, 1, 2)
            elif img.shape[1] == self.img_channels:
                pass
            else:
                raise ValueError(f"Unexpected channel dimension in img {img.shape}")
        elif img.ndim == 3:
            if img.shape[-1] == self.img_channels:
                img = img.permute(2, 0, 1).unsqueeze(0)
            elif img.shape[0] == self.img_channels:
                img = img.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected img tensor shape {img.shape}")
        else:
            raise ValueError(f"Unsupported img tensor ndim {img.ndim}")

        # Process depth
        x_img = self.cnn_img(img)
        x_img = x_img.reshape(B, -1)  # handle non-contiguous tensors
        img_feats = self.fc_img(x_img)

        # Tool and goal masks
        tm = observations[self.tool_key]
        gm = observations[self.goal_key]
        if tm.ndim == 3:
            tm = tm.unsqueeze(1)
        if gm.ndim == 3:
            gm = gm.unsqueeze(1)
        assert tm.shape[2:] == gm.shape[2:], \
            f"tool_mask {tm.shape[2:]} and goal_mask {gm.shape[2:]} must match"
        mask_input = torch.cat([tm, gm], dim=1)
        x_mask = self.cnn_mask(mask_input)
        x_mask = x_mask.reshape(B, -1)
        mask_feats = self.fc_mask(x_mask)

        # Other observations
        if self.mlp_other:
            others = [observations[k].view(B, -1) for k in self.other_keys]
            other_cat = torch.cat(others, dim=1)
            other_feats = self.mlp_other(other_cat)
        else:
            other_feats = torch.zeros(B, 0, device=mask_feats.device)

        # Final concat & project
        out = torch.cat([img_feats, mask_feats, other_feats], dim=1)
        return self.projection(out)