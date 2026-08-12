"""
STAGE 1 (CORRECTED): DCCRN (frozen teacher) -> Large IncepTCN (student)
Response-based KD only.

CORRECTED from the earlier version of this file: that version used a
fixed non-overlapping [B,1,60,320] reshape of a 19200-sample chunk, which
does not match your actual original framing convention (utils.py's
sliceframe/OverlapAndAdd: frame_size=320, frame_shift=160, 50% overlap).

This version:
    - Uses a fixed segment length of 16000 samples (1 second at 16kHz),
      per your instruction to keep 16000, not 19200.
    - Frames each segment via frame_signal() (ola.py) into overlapping
      320-sample frames with 160-sample shift -> nframes = 16000//160 = 100.
      Model input/output shape is therefore [B, 1, 100, 320] instead of the
      earlier [B, 1, 60, 320]. This is fine architecturally: every conv
      layer in PROP_TCNN uses stride (1, x) -- the frame-count axis is
      never downsampled, so the model works identically regardless of
      whether T=60 or T=100.
    - Reconstructs the model's framed output back into a real waveform via
      OverlapAdd (ola.py) -- a true, differentiable overlap-add, not a
      plain .reshape(). This matters because frames now genuinely overlap
      (unlike the old non-overlapping grid), so summing overlapping
      regions and normalizing by coverage count is required to get a
      correct waveform back, and it must stay differentiable so gradients
      from the response-based KD loss can flow back through it into the
      student model during training.
"""

import os
import builtins
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader

from losses import MultiResolutionSTFTLoss, si_sdr_loss
from ola import frame_signal, OverlapAdd
from model_2inceptcn import PROP_TCNN

from asteroid.models import DCCRNet

try:
    from pesq import pesq
    _HAS_PESQ = True
except ImportError:
    _HAS_PESQ = False

try:
    from pystoi import stoi
    _HAS_STOI = True
except ImportError:
    _HAS_STOI = False

if not (_HAS_PESQ and _HAS_STOI):
    print(
        "NOTE: PESQ and/or STOI packages are not installed. "
        "Run: pip install pesq pystoi\n"
        "Periodic eval will skip whichever metric is missing.\n"
    )


# --- 1. CONFIG ---
SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000          # 1 second, fixed segment length (was 19200 -- corrected per your instruction)
FRAME_SIZE = 320
FRAME_SHIFT = 160               # 50% overlap, matches your original sliceframe() convention
NFRAMES = TOTAL_SAMPLES // FRAME_SHIFT  # = 100
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
KD_WEIGHT = 0.2
CHECKPOINT_DIR = "checkpoint_stage1"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- periodic held-out eval config ---
TEST_NOISY_DIR = r"D:\voicebank_extracted\noisy_testset_wav"
TEST_CLEAN_DIR = r"D:\voicebank_extracted\clean_testset_wav"
EVAL_EVERY = 10          # run PESQ/STOI/SI-SDR eval every N epochs
EVAL_N_SAMPLES = 20      # how many held-out files to score each time (keeps eval fast)
PESQ_MODE = "wb" if SAMPLE_RATE == 16000 else "nb"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Segment length: {TOTAL_SAMPLES} samples | frame_size={FRAME_SIZE} | "
      f"frame_shift={FRAME_SHIFT} | nframes={NFRAMES}")


# --- 2. DATA ---

def fix_length(x, size):
    if len(x) > size:
        return x[:size]
    return np.pad(x, (0, size - len(x)))


class PairedWaveDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, total_samples=TOTAL_SAMPLES):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.total_samples = total_samples

        noisy_files = sorted(f for f in os.listdir(noisy_dir) if f.lower().endswith(".wav"))
        clean_files = sorted(f for f in os.listdir(clean_dir) if f.lower().endswith(".wav"))
        n = builtins.min(len(noisy_files), len(clean_files))
        self.files = list(zip(noisy_files[:n], clean_files[:n]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy_name, clean_name = self.files[idx]

        sr_n, noisy = wavfile.read(os.path.join(self.noisy_dir, noisy_name))
        sr_c, clean = wavfile.read(os.path.join(self.clean_dir, clean_name))
        assert sr_n == SAMPLE_RATE and sr_c == SAMPLE_RATE, (
            f"Expected {SAMPLE_RATE} Hz, got noisy={sr_n}, clean={sr_c}."
        )

        noisy = noisy.astype(np.float32) / 32768.0
        clean = clean.astype(np.float32) / 32768.0

        noisy = fix_length(noisy, self.total_samples)
        clean = fix_length(clean, self.total_samples)

        noisy = noisy / (np.max(np.abs(noisy)) + 1e-8)
        clean = clean / (np.max(np.abs(clean)) + 1e-8)

        noisy_wave = torch.tensor(noisy, dtype=torch.float32)   # (T,) raw waveform
        clean_wave = torch.tensor(clean, dtype=torch.float32)   # (T,) raw waveform

        return noisy_wave, clean_wave


# --- 3. TASK LOSS ---

def combined_loss(output, target):
    mse = nn.MSELoss()(output, target)
    cosine = 1 - torch.mean(
        torch.sum(output * target, dim=-1) /
        (torch.norm(output, dim=-1) * torch.norm(target, dim=-1) + 1e-8)
    )
    return mse + 0.1 * cosine


# --- 3b. PERIODIC HELD-OUT EVAL (SI-SDR + PESQ + STOI) ---
# Reuses the same logic as eval_stage1.py, but inline so it can run
# automatically during training on the held-out test set (never used for
# gradient updates), without needing a separate manual script call.

def run_eval(model, overlap_add, epoch_label, n_samples=EVAL_N_SAMPLES):
    was_training = model.training
    model.eval()  # switch BatchNorm to running_var/running_mean, like real inference

    eval_dataset = PairedWaveDataset(TEST_NOISY_DIR, TEST_CLEAN_DIR, total_samples=TOTAL_SAMPLES)
    n_eval = builtins.min(n_samples, len(eval_dataset))

    si_sdr_vals, pesq_vals, stoi_vals = [], [], []

    with torch.no_grad():
        for i in range(n_eval):
            noisy_wave, clean_wave = eval_dataset[i]
            noisy_wave = noisy_wave.unsqueeze(0).to(device)
            clean_wave = clean_wave.unsqueeze(0).to(device)

            noisy_grid = frame_signal(noisy_wave, FRAME_SIZE, FRAME_SHIFT).unsqueeze(1)
            out_grid = model(noisy_grid)
            out_wave = overlap_add(out_grid.squeeze(1))

            ml = builtins.min(out_wave.shape[-1], clean_wave.shape[-1])
            out_aligned = out_wave[..., :ml]
            clean_aligned = clean_wave[..., :ml]

            si_sdr_vals.append((-si_sdr_loss(out_aligned, clean_aligned)).item())

            out_np = out_aligned.squeeze(0).cpu().numpy()
            clean_np = clean_aligned.squeeze(0).cpu().numpy()

            if _HAS_PESQ:
                try:
                    pesq_vals.append(pesq(SAMPLE_RATE, clean_np, out_np, PESQ_MODE))
                except Exception:
                    pass  # skip degenerate segments rather than crashing training

            if _HAS_STOI:
                try:
                    stoi_vals.append(stoi(clean_np, out_np, SAMPLE_RATE, extended=False))
                except Exception:
                    pass

    si_sdr_arr = np.array(si_sdr_vals)
    msg = (
        f"[EVAL @ {epoch_label}] held-out ({n_eval} samples) | "
        f"SI-SDR: {si_sdr_arr.mean():.3f} dB (std {si_sdr_arr.std():.3f})"
    )
    if pesq_vals:
        p = np.array(pesq_vals)
        msg += f" | PESQ: {p.mean():.3f} (std {p.std():.3f}, n={len(p)})"
    elif _HAS_PESQ:
        msg += " | PESQ: no samples scored"
    else:
        msg += " | PESQ: skipped (not installed)"

    if stoi_vals:
        s = np.array(stoi_vals)
        msg += f" | STOI: {s.mean():.3f} (std {s.std():.3f}, n={len(s)})"
    elif _HAS_STOI:
        msg += " | STOI: no samples scored"
    else:
        msg += " | STOI: skipped (not installed)"

    print(msg)

    if was_training:
        model.train()  # restore training mode before resuming the loop


# --- 4. TRAIN ---

def train():
    noisy_dir = r"D:\voicebank_extracted\noisy_trainset_28spk_wav"
    clean_dir = r"D:\voicebank_extracted\clean_trainset_28spk_wav"

    dataset = PairedWaveDataset(noisy_dir, clean_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} training pairs.")

    student = PROP_TCNN(mode="full").to(device)
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)

    print("Loading frozen DCCRN teacher (JorisCos/DCCRNet_Libri1Mix_enhsingle_16k)...")
    teacher = DCCRNet.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    mrstft = MultiResolutionSTFTLoss().to(device)
    overlap_add = OverlapAdd(sig_len=TOTAL_SAMPLES, frame_size=FRAME_SIZE, frame_shift=FRAME_SHIFT).to(device)

    prev_epoch_kd_loss = None

    for epoch in range(EPOCHS):
        student.train()
        epoch_task_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_si_sdr = 0.0
        t0 = time.time()

        for noisy_wave, clean_wave in loader:
            noisy_wave = noisy_wave.to(device)   # (B, 16000)
            clean_wave = clean_wave.to(device)   # (B, 16000)

            opt.zero_grad()

            noisy_grid = frame_signal(noisy_wave, FRAME_SIZE, FRAME_SHIFT).unsqueeze(1)  # (B,1,100,320)
            clean_grid = frame_signal(clean_wave, FRAME_SIZE, FRAME_SHIFT).unsqueeze(1)  # (B,1,100,320)

            student_out_grid = student(noisy_grid)                      # (B,1,100,320)

            task_loss = combined_loss(student_out_grid, clean_grid)

            student_waveform = overlap_add(student_out_grid.squeeze(1))  # (B, 16000)

            with torch.no_grad():
                teacher_waveform = teacher(noisy_wave)
                if teacher_waveform.dim() == 3:
                    teacher_waveform = teacher_waveform.squeeze(1)

            ml = builtins.min(student_waveform.shape[-1], teacher_waveform.shape[-1])
            student_wave_aligned = student_waveform[..., :ml]
            teacher_wave_aligned = teacher_waveform[..., :ml].detach()

            sc_loss, mag_loss = mrstft(student_wave_aligned, teacher_wave_aligned)
            kd_loss = sc_loss + mag_loss

            total_loss = task_loss + KD_WEIGHT * kd_loss
            total_loss.backward()
            # Safety net alongside the res_scale fix in ResBlock: caps any
            # remaining gradient spikes before they can push dec.0 (or any
            # other layer) back toward the same kind of blow-up.
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            opt.step()

            epoch_task_loss += task_loss.item()
            epoch_kd_loss += kd_loss.item()
            with torch.no_grad():
                clean_aligned = clean_wave[..., :ml]
                epoch_si_sdr += (-si_sdr_loss(student_wave_aligned, clean_aligned)).item()

        scheduler.step()
        n_batches = len(loader)
        avg_task_loss = epoch_task_loss / n_batches
        avg_kd_loss = epoch_kd_loss / n_batches
        avg_si_sdr = epoch_si_sdr / n_batches

        if prev_epoch_kd_loss is None:
            kd_signal = "warmup"
            kd_delta = 0.0
        else:
            kd_delta = prev_epoch_kd_loss - avg_kd_loss
            kd_signal = "YES" if kd_delta > 0 else "NO"

        # Watch this specifically for the first several epochs: this is the
        # exact stat that blew up to millions in the earlier (unfixed) run.
        # Healthy range should stay roughly under 100-200. If it climbs into
        # the thousands again despite the res_scale fix + grad clipping,
        # stop training and investigate further before continuing.
        dec0_running_var_max = student.dec[0][1].running_var.max().item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"task_loss: {avg_task_loss:.6f} | "
            f"kd_loss: {avg_kd_loss:.6f} | "
            f"KD_signal: {kd_signal} (Δ={kd_delta:.6f}) | "
            f"SI-SDR: {avg_si_sdr:.3f} dB | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"dec.0_running_var_max: {dec0_running_var_max:.2f} | "
            f"time: {time.time()-t0:.1f}s"
        )
        prev_epoch_kd_loss = avg_kd_loss

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"large_inceptcn_epoch{epoch+1}.pt")
            torch.save(student.state_dict(), ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

        if (epoch + 1) % EVAL_EVERY == 0:
            run_eval(student, overlap_add, epoch_label=f"epoch {epoch+1}")

    final_path = os.path.join(CHECKPOINT_DIR, "large_inceptcn_final.pt")
    torch.save(student.state_dict(), final_path)
    print(f"Stage 1 complete. Final large-IncepTCN checkpoint: {final_path}")

    # Final held-out eval on the finished model, in case EPOCHS isn't a
    # multiple of EVAL_EVERY (so you always get a reading on the last epoch).
    run_eval(student, overlap_add, epoch_label="FINAL")


if __name__ == "__main__":
    train()
