"""
Eval-mode sanity check for the Stage 1 large-IncepTCN checkpoint.

Purpose: training-time SI-SDR (in distill_stage_new.py's log) is computed
with the model in .train() mode, where BatchNorm normalizes using each
batch's OWN statistics -- not running_var/running_mean. At inference
(.eval() mode), BatchNorm switches to using the stored running_var /
running_mean instead. Since dec.0's running_var_max climbed to ~28000 by
epoch 5 (vs the ~100-200 healthy range), this script checks whether that
inflated running_var actually degrades real inference quality, or whether
it's a red herring.

Run this AFTER distill_stage_new.py has produced checkpoint_stage1/large_inceptcn_final.pt.

Usage:
    python eval_stage1.py --n_samples 20
"""

import os
import argparse

import numpy as np
import torch

from distill_stage_new import PairedWaveDataset, SAMPLE_RATE, TOTAL_SAMPLES, FRAME_SIZE, FRAME_SHIFT
from ola import frame_signal, OverlapAdd
from losses import si_sdr_loss
from model_2inceptcn import PROP_TCNN

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
        "Continuing without the missing metric(s).\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join("checkpoint_stage1", "large_inceptcn_final.pt"))
    parser.add_argument("--noisy_dir", default=r"D:\voicebank_extracted\noisy_testset_wav")
    parser.add_argument("--clean_dir", default=r"D:\voicebank_extracted\clean_testset_wav")
    parser.add_argument("--n_samples", type=int, default=20,
                         help="How many individual files to evaluate one-by-one.")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load model in EVAL mode ---
    model = PROP_TCNN(mode="full").to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()  # <-- critical: switches BatchNorm to use running_var/running_mean

    # Report the exact stat we're worried about, straight from the loaded checkpoint.
    dec0_running_var_max = model.dec[0][1].running_var.max().item()
    dec0_running_var_mean = model.dec[0][1].running_var.mean().item()
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"dec.0 BatchNorm running_var: max={dec0_running_var_max:.2f}, mean={dec0_running_var_mean:.2f}")
    print(f"(For reference, your training run logged this climbing to ~27947 by epoch 5.)\n")

    # --- data ---
    dataset = PairedWaveDataset(args.noisy_dir, args.clean_dir, total_samples=TOTAL_SAMPLES)
    n_eval = min(args.n_samples, len(dataset))
    print(f"Evaluating on {n_eval} samples (out of {len(dataset)} total in this dir).\n")

    overlap_add = OverlapAdd(sig_len=TOTAL_SAMPLES, frame_size=FRAME_SIZE, frame_shift=FRAME_SHIFT).to(device)

    per_sample_si_sdr = []
    per_sample_pesq = []
    per_sample_stoi = []

    # PESQ's 'wb' (wideband) mode requires 16000 Hz input -- matches SAMPLE_RATE here.
    pesq_mode = "wb" if SAMPLE_RATE == 16000 else "nb"

    with torch.no_grad():
        for i in range(n_eval):
            noisy_wave, clean_wave = dataset[i]
            noisy_wave = noisy_wave.unsqueeze(0).to(device)   # (1, T)
            clean_wave = clean_wave.unsqueeze(0).to(device)   # (1, T)

            noisy_grid = frame_signal(noisy_wave, FRAME_SIZE, FRAME_SHIFT).unsqueeze(1)  # (1,1,100,320)

            out_grid = model(noisy_grid)                       # eval-mode forward pass
            out_wave = overlap_add(out_grid.squeeze(1))         # (1, T)

            ml = min(out_wave.shape[-1], clean_wave.shape[-1])
            out_aligned = out_wave[..., :ml]
            clean_aligned = clean_wave[..., :ml]

            sdr = (-si_sdr_loss(out_aligned, clean_aligned)).item()
            per_sample_si_sdr.append(sdr)

            # PESQ and STOI both expect 1-D numpy arrays on CPU.
            out_np = out_aligned.squeeze(0).cpu().numpy()
            clean_np = clean_aligned.squeeze(0).cpu().numpy()

            pesq_val = None
            if _HAS_PESQ:
                try:
                    pesq_val = pesq(SAMPLE_RATE, clean_np, out_np, pesq_mode)
                    per_sample_pesq.append(pesq_val)
                except Exception as e:
                    # PESQ can fail on near-silent or degenerate segments -- skip those
                    # rather than crashing the whole eval run.
                    print(f"    [sample {i}] PESQ failed: {e}")

            stoi_val = None
            if _HAS_STOI:
                try:
                    stoi_val = stoi(clean_np, out_np, SAMPLE_RATE, extended=False)
                    per_sample_stoi.append(stoi_val)
                except Exception as e:
                    print(f"    [sample {i}] STOI failed: {e}")

            pesq_str = f"{pesq_val:5.3f}" if pesq_val is not None else "  n/a"
            stoi_str = f"{stoi_val:5.3f}" if stoi_val is not None else "  n/a"
            print(f"  sample {i:3d}: SI-SDR = {sdr:7.3f} dB | PESQ = {pesq_str} | STOI = {stoi_str}")

    per_sample_si_sdr = np.array(per_sample_si_sdr)
    print("\n--- Summary ---")
    print(f"Mean eval-mode SI-SDR: {per_sample_si_sdr.mean():.3f} dB "
          f"(std {per_sample_si_sdr.std():.3f}, min {per_sample_si_sdr.min():.3f}, "
          f"max {per_sample_si_sdr.max():.3f})")

    if per_sample_pesq:
        arr = np.array(per_sample_pesq)
        print(f"Mean PESQ (wb, 1.0-4.5 scale): {arr.mean():.3f} "
              f"(std {arr.std():.3f}, min {arr.min():.3f}, max {arr.max():.3f}) "
              f"[{len(arr)}/{n_eval} samples scored]")
    elif _HAS_PESQ:
        print("PESQ: no samples scored successfully.")
    else:
        print("PESQ: skipped (package not installed -- pip install pesq)")

    if per_sample_stoi:
        arr = np.array(per_sample_stoi)
        print(f"Mean STOI (0-1 scale):         {arr.mean():.3f} "
              f"(std {arr.std():.3f}, min {arr.min():.3f}, max {arr.max():.3f}) "
              f"[{len(arr)}/{n_eval} samples scored]")
    elif _HAS_STOI:
        print("STOI: no samples scored successfully.")
    else:
        print("STOI: skipped (package not installed -- pip install pystoi)")

    print(f"\n(Compare SI-SDR to the LAST logged TRAIN-mode SI-SDR from epoch 5: 6.255 dB.")
    print(f"If eval-mode mean is noticeably lower or wildly more variable,")
    print(f"that would point to the inflated running_var hurting real inference quality --")
    print(f"in that case, lower res_scale further (e.g. 0.05) and/or lower BatchNorm")
    print(f"momentum (e.g. momentum=0.01) in ResBlock's TCM_net, then retrain.")
    print(f"For PESQ/STOI there's no prior baseline yet from this pipeline -- these are")
    print(f"your first readings, useful mainly as a reference point for future checkpoints.)")


if __name__ == "__main__":
    main()
