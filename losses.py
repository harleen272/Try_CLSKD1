"""
Loss functions for the two-stage KD pipeline:
  Stage 1: DCCRN (frozen teacher) -> Large IncepTCN (student), response-based KD
  Stage 2: Large IncepTCN (frozen teacher) -> Compressed IncepTCN (student), feature-based KD

The MultiResolutionSTFTLoss below is functionally the same loss used in the
CLSKD reference repo's framework.py, but the internal `stft()` helper is
rewritten for modern PyTorch. The original repo's version does:

    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

This relies on torch.stft's old default (return_complex=False), which returns
a real tensor with a trailing (real, imag) dim. Recent PyTorch versions
require return_complex=True and will warn or error otherwise. Rewritten here
with .abs() on a proper complex tensor, which is numerically identical to
sqrt(real**2 + imag**2) but future-proof.
"""

import torch
import torch.nn.functional as F
from torch import nn


def stft_mag(x, fft_size, hop_size, win_length, window):
    """Magnitude spectrogram via modern complex-valued STFT.

    Args:
        x: (B, T) waveform
    Returns:
        (B, #frames, fft_size // 2 + 1) magnitude spectrogram
    """
    x_stft = torch.stft(
        x, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
        window=window, return_complex=True
    )
    mag = torch.clamp(x_stft.abs(), min=1e-7)
    return mag.transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        x_mag = stft_mag(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft_mag(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Same defaults as the CLSKD reference repo's framework.py."""

    def __init__(self,
                 fft_sizes=(1024, 2048, 512),
                 hop_sizes=(120, 240, 50),
                 win_lengths=(600, 1200, 240),
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList(
            [STFTLoss(fs, ss, wl, window) for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths)]
        )
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """
        Args:
            x: (B, T) predicted waveform
            y: (B, T) ground-truth / target waveform
        Returns:
            sc_loss, mag_loss (both already scaled by factor_sc / factor_mag)
        """
        sc_loss, mag_loss = 0.0, 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return self.factor_sc * sc_loss, self.factor_mag * mag_loss

    def total(self, x, y):
        sc, mag = self.forward(x, y)
        return sc + mag


def si_sdr_loss(estimate, target, eps=1e-8):
    """Negative SI-SDR, for use as a training loss (minimize this).
    estimate, target: (B, T) waveforms. Kept as its own function (not folded
    into MultiResolutionSTFTLoss) so you can log SI-SDR every epoch as a
    diagnostic even on runs that don't train with it directly -- this is the
    metric that silently broke for SPKD/CLSKD earlier, so keep watching it.
    """
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    alpha = (torch.sum(estimate * target, dim=-1, keepdim=True) /
              (torch.sum(target * target, dim=-1, keepdim=True) + eps))
    proj = alpha * target
    noise = estimate - proj
    ratio = (torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    si_sdr = 10 * torch.log10(ratio + eps)
    return -si_sdr.mean()  # negative so it can be minimized like any other loss


class FrameLevelSKD(nn.Module):
    """Corrected frame-level Similarity-preserving Knowledge Distillation,
    per CLSKD paper equations 6-7 (Cheng et al., Interspeech 2022).

    This is DIFFERENT from the reference repo's SPKDLoss, which flattens the
    entire feature map (all frames + channels) into one vector per sample
    before building a single batch-wide similarity matrix. That is the
    original, less-stable SPKD baseline the paper compares against -- not
    the paper's own proposed method. This class instead builds one
    similarity matrix PER TIME FRAME, which is the actual novel piece of
    CLSKD and is what's used in Stage 2 below.

    Expects features shaped (B, C, T, ...) where T is the time-frame axis
    to loop over. Any trailing spatial dims are flattened together with C
    for each frame.
    """

    def __init__(self, reduction="batchmean"):
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def _similarity_matrix(frame_feat):
        # frame_feat: (B, D) already flattened for one frame
        g = torch.matmul(frame_feat, frame_feat.t())
        return F.normalize(g, p=2, dim=1)

    def forward(self, student_feat, teacher_feat):
        """
        student_feat, teacher_feat: (B, C, T, *spatial) -- must have the
        same T (time-frame count) on both sides. If C or spatial dims
        differ between student/teacher, project one side to match BEFORE
        calling this (e.g. a 1x1 conv), since this loss only compares
        per-frame relational structure, not raw feature values, but still
        needs matching flattened dimensionality to compute a dot product.
        """
        assert student_feat.shape[0] == teacher_feat.shape[0], "batch size mismatch"
        assert student_feat.shape[2] == teacher_feat.shape[2], (
            f"time-frame count mismatch: student has {student_feat.shape[2]} frames, "
            f"teacher has {teacher_feat.shape[2]} -- these must match for frame-level SKD"
        )
        B = student_feat.shape[0]
        T = student_feat.shape[2]
        total_loss = 0.0
        for t in range(T):
            s_frame = torch.flatten(student_feat[:, :, t, ...], 1)  # (B, D_s)
            t_frame = torch.flatten(teacher_feat[:, :, t, ...], 1)  # (B, D_t)
            g_s = self._similarity_matrix(s_frame)
            g_t = self._similarity_matrix(t_frame)
            total_loss = total_loss + torch.norm(g_t - g_s, p="fro") ** 2
        if self.reduction == "batchmean":
            return total_loss / (B ** 2)
        return total_loss
