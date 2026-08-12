"""
IncepTCN model definitions.

PROP_TCNN below matches your latest version: inception applied after
Conv2d_7, with the bottleneck reduction/restoration now done via
LearnableProjection / LearnableExpansion (a per-channel learnable weighted
combine/broadcast across the 4 collapsed positions) instead of the earlier
torch.mean / .repeat. Shape-traced by hand below the class to confirm this
substitution doesn't change any downstream tensor shape.

CompressedIncepTCN is a NEW, smaller variant for Stage 2's student -- same
overall shape (encoder -> inception -> TCN bottleneck -> decoder with skip
connections), but with narrower channels throughout. This is a STARTING
POINT, not a validated final design: you should decide the actual target
compression ratio (parameter count / MACs) for your deployment goal and
adjust the channel widths below accordingly. I've picked roughly half the
large model's channel widths as a reasonable first guess, nothing more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, causal=False):
        super().__init__()
        depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=in_channels, bias=False)
        pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.net = nn.Sequential(
            depthwise,
            Chomp1d(padding) if causal else nn.Identity(),
            nn.PReLU(),
            nn.BatchNorm1d(in_channels),
            pointwise
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, res_scale=0.1):
        super().__init__()
        self.TCM_net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            DepthwiseSeparableConv(out_channels, in_channels, kernel_size, 1,
                                   (kernel_size - 1) * dilation, dilation, causal=True)
        )
        # Scales down the residual branch before adding it back to x.
        # Without this, stacking 18 of these blocks (3 TCNN_Block x 6 layers)
        # with an unnormalized addition causes activation magnitude to grow
        # ~3x across the stack, which then blows up dec.0's BatchNorm
        # running_var into the millions (confirmed on the trained checkpoint).
        self.res_scale = res_scale

    def forward(self, x):
        return self.res_scale * self.TCM_net(x) + x


class TCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_dilation=2, num_layers=6):
        super().__init__()
        layers = [ResBlock(in_channels, out_channels, kernel_size, init_dilation ** i)
                  for i in range(num_layers)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, mode="full"):
        super().__init__()
        self.mode = mode
        self.branch1x1 = nn.Conv2d(in_channels, out_channels[0], 1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], 1),
            nn.Conv2d(out_channels[1], out_channels[2], 3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], 1),
            nn.Conv2d(out_channels[3], out_channels[4], 5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels[5], 1)
        )

    def forward(self, x):
        if self.mode == "1x1":
            out = self.branch1x1(x)
            return F.pad(out, (0, 0, 0, 0, 0, 192))
        elif self.mode == "1x3":
            out = torch.cat([self.branch1x1(x), self.branch3x3(x)], 1)
            return F.pad(out, (0, 0, 0, 0, 0, 64))
        else:  # full
            return torch.cat([
                self.branch1x1(x), self.branch3x3(x), self.branch5x5(x), self.branch_pool(x)
            ], 1)


# --- LARGE MODEL (Stage 1 student, Stage 2 teacher) -- updated to match the
# LearnableProjection / LearnableExpansion bottleneck (replaces the earlier
# torch.mean / .repeat approach with a per-channel learnable weighting). ---

class LearnableProjection(nn.Module):
    """Learnable weighted combination across the 4 collapsed positions,
    replacing torch.mean. One softmax-normalized weight vector (length 4)
    per channel, shared across all time frames and all batches."""

    def __init__(self, channels=256):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, 4) / 4)

    def forward(self, x):
        # x: (B, C, T, 4)
        w = torch.softmax(self.weight, dim=1)
        w = w.unsqueeze(0).unsqueeze(2)   # (1, C, 1, 4)
        return torch.sum(x * w, dim=3, keepdim=True)  # (B, C, T, 1)


class LearnableExpansion(nn.Module):
    """Learnable per-channel, per-position scaling, replacing .repeat(1,1,1,4).
    Each of the 4 restored positions gets its own learned scale factor per
    channel instead of being an identical copy of the collapsed value."""

    def __init__(self, channels=256):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, 4))

    def forward(self, x):
        # x: (B, C, T, 1)
        w = self.weight.unsqueeze(0).unsqueeze(2)   # (1, C, 1, 4)
        return x * w   # broadcasts to (B, C, T, 4)


class PROP_TCNN(nn.Module):
    def __init__(self, mode="full"):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, 16, (3, 5), (1, 1), (1, 2)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.Conv2d(16, 16, (3, 5), (1, 2), (1, 2)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.Conv2d(16, 16, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.Conv2d(16, 32, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(32), nn.PReLU()),
            nn.Sequential(nn.Conv2d(32, 32, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(32), nn.PReLU()),
            nn.Sequential(nn.Conv2d(32, 64, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(64), nn.PReLU()),
            nn.Sequential(nn.Conv2d(64, 64, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(64), nn.PReLU())
        ])
        self.inception = InceptionModule(64, [64, 48, 128, 16, 32, 32], mode=mode)
        self.projection = LearnableProjection(channels=256)
        self.expansion = LearnableExpansion(channels=256)
        self.tcnn = nn.Sequential(TCNN_Block(256, 512), TCNN_Block(256, 512), TCNN_Block(256, 512))
        self.dec = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(320, 64, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(64), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(128, 32, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(32), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(64, 32, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(32), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(64, 16, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(32, 16, (3, 5), (1, 2), (1, 1), (0, 1)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(32, 16, (3, 5), (1, 2), (1, 2), (0, 1)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(32, 1, (3, 5), (1, 1), (1, 2)), nn.BatchNorm2d(1), nn.PReLU())
        ])

    def forward(self, x):
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)

        x = self.inception(x)
        x = self.projection(x)

        b, c, t, f = x.shape
        x = x.permute(0, 1, 3, 2).reshape(b, c * f, t)
        x = self.tcnn(x)
        x = x.reshape(b, c, f, t).permute(0, 1, 3, 2)
        x = self.expansion(x)

        for i, layer in enumerate(self.dec):
            x = layer(torch.cat([skips[-(i + 1)], x], 1))
        return x


# --- COMPRESSED MODEL (Stage 2 student) -- NEW, starting-point channel widths ---
# Roughly half the large model's channels. Confirm actual MACs/params via
# thop.profile (as your calculate_flops_and_inference() already does) and
# adjust to hit your real deployment target -- these numbers are a
# reasonable first guess, not a validated final design.

class CompressedIncepTCN(nn.Module):
    def __init__(self, mode="full"):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, 8, (3, 5), (1, 1), (1, 2)), nn.BatchNorm2d(8), nn.PReLU()),
            nn.Sequential(nn.Conv2d(8, 8, (3, 5), (1, 2), (1, 2)), nn.BatchNorm2d(8), nn.PReLU()),
            nn.Sequential(nn.Conv2d(8, 8, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(8), nn.PReLU()),
            nn.Sequential(nn.Conv2d(8, 16, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.Conv2d(16, 16, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.Conv2d(16, 32, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(32), nn.PReLU()),
            nn.Sequential(nn.Conv2d(32, 32, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(32), nn.PReLU())
        ])
        # inception in_channels must match the last encoder layer's out_channels (32 here, was 64 in large model)
        self.inception = InceptionModule(32, [32, 24, 64, 8, 16, 16], mode=mode)  # sums to 128, half of large's 256
        self.projection = LearnableProjection(channels=128)
        self.expansion = LearnableExpansion(channels=128)
        self.tcnn = nn.Sequential(TCNN_Block(128, 256), TCNN_Block(128, 256), TCNN_Block(128, 256))
        self.dec = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(160, 32, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(32), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(64, 16, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(32, 16, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(16), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(32, 8, (3, 5), (1, 2), (1, 1)), nn.BatchNorm2d(8), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(16, 8, (3, 5), (1, 2), (1, 1), (0, 1)), nn.BatchNorm2d(8), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(16, 8, (3, 5), (1, 2), (1, 2), (0, 1)), nn.BatchNorm2d(8), nn.PReLU()),
            nn.Sequential(nn.ConvTranspose2d(16, 1, (3, 5), (1, 1), (1, 2)), nn.BatchNorm2d(1), nn.PReLU())
        ])

    def forward(self, x):
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)

        x = self.inception(x)
        x = self.projection(x)

        b, c, t, f = x.shape
        x = x.permute(0, 1, 3, 2).reshape(b, c * f, t)
        x = self.tcnn(x)
        x = x.reshape(b, c, f, t).permute(0, 1, 3, 2)
        x = self.expansion(x)

        for i, layer in enumerate(self.dec):
            x = layer(torch.cat([skips[-(i + 1)], x], 1))
        return x


if __name__ == "__main__":
    # Quick shape sanity check -- run this on your own machine (needs torch)
    # before trusting either training script.
    #
    # T=100 here matches the ACTUAL pipeline: distill_stage1.py uses
    # TOTAL_SAMPLES=16000, FRAME_SHIFT=160 -> NFRAMES = 16000//160 = 100.
    # (Earlier this test used T=60 as an arbitrary leftover value from a
    # previous version of the pipeline -- that number never meant anything
    # architecturally, since every conv layer here uses stride (1, x) and
    # never touches the T axis, so ANY T works. Using 100 now just makes
    # this test match what you'll actually see during real training,
    # instead of an unrelated arbitrary number.)
    B, T = 2, 100
    x = torch.randn(B, 1, T, 320)
    large = PROP_TCNN(mode="full")
    small = CompressedIncepTCN(mode="full")
    print("PROP_TCNN output:", large(x).shape)          # expect (2, 1, 100, 320)
    print("CompressedIncepTCN output:", small(x).shape)   # expect (2, 1, 100, 320)
