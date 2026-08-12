"""
Differentiable, PyTorch-native equivalents of utils.py's sliceframe() and
OverlapAndAdd(). Same framing convention (frame_size=320, frame_shift=160,
50% overlap, zero-padded tail), but implemented via torch.unfold/F.fold so
gradients can flow through framing -> model -> reconstruction during
training. The original utils.py versions are plain NumPy and would break
backprop if used directly inside a training loop.

Verified frame count for SAMPLE_RATE=16000, TOTAL_SAMPLES=16000 (1 second),
frame_size=320, frame_shift=160:
    nframes = 16000 // 160 = 100
    last frame starts at 99*160=15840, ends at 16160 (160 samples past the
    end of the 16000-sample signal) -> last frame is real audio for its
    first 160 samples, zero-padded for its last 160 -- exactly matching
    sliceframe()'s behavior in utils.py.
"""

import torch
import torch.nn.functional as F


def frame_signal(x, frame_size=320, frame_shift=160):
    """Differentiable equivalent of utils.py's sliceframe(), overlapping,
    zero-padded tail.

    Args:
        x: (B, T) waveform, T should equal the fixed TOTAL_SAMPLES this
           pipeline uses (e.g. 16000).
    Returns:
        (B, nframes, frame_size) framed tensor, nframes = T // frame_shift.
    """
    B, T = x.shape
    nframes = T // frame_shift
    padded_len = (nframes - 1) * frame_shift + frame_size
    pad_amount = padded_len - T
    if pad_amount > 0:
        x = F.pad(x, (0, pad_amount))
    frames = x.unfold(-1, frame_size, frame_shift)  # (B, nframes, frame_size)
    return frames


class OverlapAdd(torch.nn.Module):
    """Differentiable equivalent of utils.py's OverlapAndAdd(), implemented
    via torch.nn.functional.fold (the standard differentiable inverse of
    unfold/im2col). The per-position overlap-count normalization is
    precomputed once at construction time (it only depends on nframes/
    frame_size/frame_shift, which are fixed for this pipeline -- not on the
    actual data), then reused on every forward call.

    Must be constructed with the SAME sig_len (TOTAL_SAMPLES) used when the
    corresponding frame_signal() call produced the frames being reconstructed.
    """

    def __init__(self, sig_len, frame_size=320, frame_shift=160):
        super().__init__()
        nframes = sig_len // frame_shift
        padded_len = (nframes - 1) * frame_shift + frame_size

        self.sig_len = sig_len
        self.nframes = nframes
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.padded_len = padded_len

        # Precompute the overlap-count normalizer once: fold a tensor of
        # all-ones frames, giving "how many frames cover each sample
        # position" -- exactly what OverlapAndAdd's `ones` accumulator does.
        ones = torch.ones(1, nframes, frame_size)
        ones_cols = ones.transpose(1, 2)  # (1, frame_size, nframes) -- fold's expected "columns" layout
        folded_ones = F.fold(
            ones_cols, output_size=(1, padded_len),
            kernel_size=(1, frame_size), stride=(1, frame_shift)
        )
        norm = folded_ones.reshape(padded_len).clamp(min=1e-8)
        self.register_buffer("norm", norm)

    def forward(self, frames):
        """
        Args:
            frames: (B, nframes, frame_size)
        Returns:
            (B, sig_len) reconstructed waveform, trimmed back to the
            original fixed length (dropping the zero-padded tail region).
        """
        assert frames.shape[1] == self.nframes and frames.shape[2] == self.frame_size, (
            f"Expected frames shaped (B, {self.nframes}, {self.frame_size}), got {tuple(frames.shape)}"
        )
        B = frames.shape[0]
        cols = frames.transpose(1, 2)  # (B, frame_size, nframes)
        folded = F.fold(
            cols, output_size=(1, self.padded_len),
            kernel_size=(1, self.frame_size), stride=(1, self.frame_shift)
        )
        sig = folded.reshape(B, self.padded_len)
        sig = sig / self.norm.unsqueeze(0)
        return sig[:, :self.sig_len]


if __name__ == "__main__":
    # Sanity check: framing a signal then overlap-adding it back should
    # reconstruct the original (up to the zero-padded tail region, which
    # gets normalized/trimmed away). Run this on your own machine.
    torch.manual_seed(0)
    sig_len = 16000
    x = torch.randn(2, sig_len)
    frames = frame_signal(x, frame_size=320, frame_shift=160)
    print("frames shape:", frames.shape)  # expect (2, 100, 320)

    ola = OverlapAdd(sig_len=sig_len, frame_size=320, frame_shift=160)
    x_hat = ola(frames)
    print("reconstructed shape:", x_hat.shape)  # expect (2, 16000)
    print("max reconstruction error:", (x - x_hat).abs().max().item())  # expect ~0 (float precision only)
