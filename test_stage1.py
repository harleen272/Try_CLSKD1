import os
import csv
import argparse
import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

from ola import frame_signal, OverlapAdd
from model_1inceptcn import PROP_TCNN

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
TOTAL_SAMPLES = 16000
FRAME_SIZE = 320
FRAME_SHIFT = 160


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fix_length(x, size):

    if len(x) > size:
        return x[:size]

    return np.pad(x, (0, size - len(x)))


def load_wav_mono(path, sr_expected=SAMPLE_RATE):

    sr, x = wavfile.read(path)

    assert sr == sr_expected, (
        f"Expected {sr_expected} Hz but got {sr} Hz in {path}"
    )

    # Convert PCM16 to float
    x = x.astype(np.float32) / 32768.0

    # Keep same 1-second length as training
    x = fix_length(x, TOTAL_SAMPLES)

    # Same normalization used during training
    x = x / (np.max(np.abs(x)) + 1e-8)

    return x


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
        noisy_dir,
        clean_dir,
        ckpt,
        out_csv,
        batch_size=1,
        device=None):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")


    # --------------------------------------------------------
    # Load trained Stage-1 student
    # --------------------------------------------------------

    print("\nLoading trained IncepTCN checkpoint...")

    model = PROP_TCNN(mode="full").to(device)

    checkpoint = torch.load(
        ckpt,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print("Checkpoint loaded successfully.")


    # --------------------------------------------------------
    # Overlap Add
    # --------------------------------------------------------

    overlap_add = OverlapAdd(
        sig_len=TOTAL_SAMPLES,
        frame_size=FRAME_SIZE,
        frame_shift=FRAME_SHIFT
    ).to(device)


    # --------------------------------------------------------
    # Find test files
    # --------------------------------------------------------

    noisy_files = sorted(
        f for f in os.listdir(noisy_dir)
        if f.lower().endswith(".wav")
    )

    clean_files = sorted(
        f for f in os.listdir(clean_dir)
        if f.lower().endswith(".wav")
    )

    n = min(
        len(noisy_files),
        len(clean_files)
    )

    pairs = list(
        zip(
            noisy_files[:n],
            clean_files[:n]
        )
    )

    print(f"\nNoisy test files : {len(noisy_files)}")
    print(f"Clean test files : {len(clean_files)}")
    print(f"Test pairs       : {len(pairs)}")


    # --------------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------------

    sum_pesq = 0.0
    sum_stoi = 0.0

    valid_pesq = 0
    valid_stoi = 0

    results = []


    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    for start in tqdm(
            range(0, len(pairs), batch_size),
            desc="Evaluating"):

        batch = pairs[
            start:start + batch_size
        ]

        noisy_batch = []
        clean_batch = []
        names = []


        # ----------------------------------------------------
        # Load batch
        # ----------------------------------------------------

        for noisy_name, clean_name in batch:

            noisy_path = os.path.join(
                noisy_dir,
                noisy_name
            )

            clean_path = os.path.join(
                clean_dir,
                clean_name
            )

            noisy = load_wav_mono(
                noisy_path
            )

            clean = load_wav_mono(
                clean_path
            )

            noisy_batch.append(noisy)

            clean_batch.append(clean)

            names.append(
                (
                    noisy_name,
                    clean_name
                )
            )


        # ----------------------------------------------------
        # Convert to numpy batch
        # ----------------------------------------------------

        noisy_batch = np.stack(
            noisy_batch,
            axis=0
        )

        clean_batch = np.stack(
            clean_batch,
            axis=0
        )


        # ====================================================
        # STUDENT INFERENCE
        # ====================================================

        with torch.no_grad():

            noisy_tensor = torch.from_numpy(
                noisy_batch
            ).to(device)


            # -----------------------------------------------
            # Frame waveform
            #
            # (B,16000)
            #       ↓
            # (B,100,320)
            #       ↓
            # (B,1,100,320)
            # -----------------------------------------------

            noisy_frames = frame_signal(
                noisy_tensor,
                FRAME_SIZE,
                FRAME_SHIFT
            ).unsqueeze(1)


            # -----------------------------------------------
            # IncepTCN enhancement
            # -----------------------------------------------

            enhanced_frames = model(
                noisy_frames
            )


            # -----------------------------------------------
            # Reconstruct waveform using OverlapAdd
            #
            # (B,1,100,320)
            #        ↓
            # (B,16000)
            # -----------------------------------------------

            enhanced_wave = overlap_add(
                enhanced_frames.squeeze(1)
            )


            enhanced_wave = (
                enhanced_wave
                .cpu()
                .numpy()
            )


        # ====================================================
        # PESQ + STOI
        # ====================================================

        for idx in range(
                enhanced_wave.shape[0]):

            ref = clean_batch[idx]

            deg = enhanced_wave[idx]


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Do NOT normalize enhanced waveform again here.
            #
            # We evaluate the model output as produced.
            # ------------------------------------------------


            # ------------------------------------------------
            # Match lengths
            # ------------------------------------------------

            ml = min(
                len(ref),
                len(deg)
            )

            ref = ref[:ml]

            deg = deg[:ml]


            # ------------------------------------------------
            # Convert to float32 and prevent invalid amplitude
            # ------------------------------------------------

            ref = np.asarray(
                ref,
                dtype=np.float32
            )

            deg = np.asarray(
                deg,
                dtype=np.float32
            )


            ref = np.clip(
                ref,
                -1.0,
                1.0
            )

            deg = np.clip(
                deg,
                -1.0,
                1.0
            )


            # =================================================
            # PESQ
            # =================================================

            try:

                p = pesq(
                    SAMPLE_RATE,
                    ref,
                    deg,
                    "wb"
                )

                sum_pesq += p

                valid_pesq += 1


            except Exception as e:

                print(
                    f"\nPESQ failed for "
                    f"{names[idx][0]} : {e}"
                )

                p = np.nan


            # =================================================
            # STOI
            # =================================================

            try:

                s = stoi(
                    ref,
                    deg,
                    SAMPLE_RATE,
                    extended=False
                )

                sum_stoi += s

                valid_stoi += 1


            except Exception as e:

                print(
                    f"\nSTOI failed for "
                    f"{names[idx][0]} : {e}"
                )

                s = np.nan


            # ------------------------------------------------
            # Store individual file results
            # ------------------------------------------------

            results.append({

                "noisy":
                    names[idx][0],

                "clean":
                    names[idx][1],

                "pesq":
                    p,

                "stoi":
                    s

            })


    # ========================================================
    # FINAL AVERAGES
    # ========================================================

    if valid_pesq > 0:

        avg_pesq = (
            sum_pesq /
            valid_pesq
        )

    else:

        avg_pesq = np.nan


    if valid_stoi > 0:

        avg_stoi = (
            sum_stoi /
            valid_stoi
        )

    else:

        avg_stoi = np.nan


    # ========================================================
    # SAVE RESULTS TO CSV
    # ========================================================

    with open(
            out_csv,
            "w",
            newline="",
            encoding="utf-8") as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[

                "noisy",
                "clean",
                "pesq",
                "stoi"

            ]
        )

        writer.writeheader()

        for row in results:

            writer.writerow(row)


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")

    print("=" * 60)

    print(
        "STAGE-1 IncepTCN EVALUATION"
    )

    print("=" * 60)

    print(
        f"Total test files : "
        f"{len(pairs)}"
    )

    print(
        f"Valid PESQ files : "
        f"{valid_pesq}"
    )

    print(
        f"Valid STOI files : "
        f"{valid_stoi}"
    )

    print("-" * 60)

    print(
        f"Average PESQ     : "
        f"{avg_pesq:.4f}"
    )

    print(
        f"Average STOI     : "
        f"{avg_stoi:.4f}"
    )

    print("=" * 60)

    print(
        f"Per-file results saved to:"
        f"\n{out_csv}"
    )

    print("=" * 60)

    # -------------------------------------------------------
    # Save a small Markdown summary for easy sharing/viewing
    # -------------------------------------------------------

    try:
        report_md = out_csv.rsplit('.', 1)[0] + "_summary.md"
        with open(report_md, "w", encoding="utf-8") as rf:
            rf.write("# Stage-1 IncepTCN Evaluation Summary\n\n")
            rf.write(f"- Total test files: {len(pairs)}\n")
            rf.write(f"- Valid PESQ files: {valid_pesq}\n")
            rf.write(f"- Valid STOI files: {valid_stoi}\n")
            rf.write(f"- Average PESQ: {avg_pesq:.4f}\n")
            rf.write(f"- Average STOI: {avg_stoi:.4f}\n\n")
            rf.write(f"Per-file results CSV: {out_csv}\n")

        print(f"Summary saved to: {report_md}")
    except Exception as e:
        print(f"Failed to write summary markdown: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained Stage-1 "
            "IncepTCN using PESQ and STOI"
        )
    )


    parser.add_argument(
        "--ckpt",
        default=r"D:\try_clskd\checkpoint_stage1\large_inceptcn_final.pt"
    )
    parser.add_argument(
        "--noisy_dir",
        default=r"D:\voicebank_extracted\noisy_testset_wav"
    )
    parser.add_argument(
        "--clean_dir",
        default=r"D:\voicebank_extracted\clean_testset_wav"
    )


    parser.add_argument(

        "--out_csv",

        default="eval_results.csv",

        help=(
            "CSV file for individual "
            "PESQ/STOI scores"
        )

    )


    parser.add_argument(

        "--batch_size",

        type=int,

        default=1,

        help="Evaluation batch size"

    )


    args = parser.parse_args()


    evaluate(

        noisy_dir=
            args.noisy_dir,

        clean_dir=
            args.clean_dir,

        ckpt=
            args.ckpt,

        out_csv=
            args.out_csv,

        batch_size=
            args.batch_size

    )