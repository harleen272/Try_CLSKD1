""" 
STAGE 2 (CORRECTED): Large IncepTCN (frozen, from Stage 1) -> Compressed 
IncepTCN (final student). 
 
Same correction as distill_stage1.py: fixed 16000-sample segments, framed 
via frame_signal() (frame_size=320, frame_shift=160, 50% overlap) into 
[B,1,100,320], with true differentiable overlap-add reconstruction (ola.py) 
instead of a plain .reshape() wherever a waveform-level comparison is 
needed (the optional response-based KD term and the SI-SDR diagnostic). 
 
Uses: 
  - Task loss: compressed model's framed output vs. ground-truth clean 
    framed target (same combined_loss as Stage 1). 
  - Feature-based KD: FrameLevelSKD (losses.py) on the TCN bottleneck 
    output of both models, via forward hooks on self.tcnn. 
  - Optional response-based KD: compressed output vs. large-IncepTCN 
    output, both reconstructed to real waveforms via OverlapAdd before 
    comparison with MultiResolutionSTFTLoss. 
""" 
 
import os 
import time 
 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
 
from losses import MultiResolutionSTFTLoss, FrameLevelSKD, si_sdr_loss 
from ola import frame_signal, OverlapAdd 
from distill_stage_new import ( 
    PairedWaveDataset, combined_loss, 
    SAMPLE_RATE, TOTAL_SAMPLES, FRAME_SIZE, FRAME_SHIFT, NFRAMES, 
) 
from model_2inceptcn import PROP_TCNN, CompressedIncepTCN 
 
 
# --- 1. CONFIG --- 
EPOCHS = 5 
BATCH_SIZE = 16 
LR = 1e-3 
FEATURE_KD_WEIGHT = 0.5 
RESPONSE_KD_WEIGHT = 0.1 
LARGE_CKPT = os.path.join("checkpoint_stage1", "large_inceptcn_final.pt") 
CHECKPOINT_DIR = "checkpoint_stage2" 
os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}") 
 
 
# --- 2. BOTTLENECK FEATURE HOOKS --- 
 
class BottleneckGrabber: 
    def __init__(self, model): 
        self.feat = None 
        self.handle = model.tcnn.register_forward_hook(self._hook) 
 
    def _hook(self, module, inp, out): 
        self.feat = out 
 
    def remove(self): 
        self.handle.remove() 
 
 
# --- 3. PROJECTION FOR CHANNEL MISMATCH (built lazily once shapes are known) --- 
 
class LazyProjection(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.proj = None 
 
    def build_if_needed(self, in_channels, out_channels, device): 
        if self.proj is None: 
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1).to(device) 
            print(f"  [LazyProjection] built Conv1d({in_channels} -> {out_channels}) " 
                  f"to match student bottleneck channels to teacher's for the SKD loss.") 
 
    def forward(self, x): 
        return self.proj(x) 
 
 
def train(): 
    noisy_dir = r"D:\voicebank_extracted\noisy_trainset_28spk_wav" 
    clean_dir = r"D:\voicebank_extracted\clean_trainset_28spk_wav" 
 
    dataset = PairedWaveDataset(noisy_dir, clean_dir) 
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) 
    print(f"Loaded {len(dataset)} training pairs.") 
 
    teacher = PROP_TCNN(mode="full").to(device) 
    teacher.load_state_dict(torch.load(LARGE_CKPT, map_location=device)) 
    teacher.eval() 
    for p in teacher.parameters(): 
        p.requires_grad = False 
    print(f"Loaded frozen large-IncepTCN teacher from {LARGE_CKPT}") 
 
    student = CompressedIncepTCN(mode="full").to(device) 
    opt = torch.optim.Adam(student.parameters(), lr=LR) 
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5) 
 
    teacher_grabber = BottleneckGrabber(teacher) 
    student_grabber = BottleneckGrabber(student) 
 
    skd_loss_fn = FrameLevelSKD(reduction="batchmean") 
    mrstft = MultiResolutionSTFTLoss().to(device) 
    overlap_add = OverlapAdd(sig_len=TOTAL_SAMPLES, frame_size=FRAME_SIZE, frame_shift=FRAME_SHIFT).to(device) 
    projection = LazyProjection() 
 
    for epoch in range(EPOCHS): 
        student.train() 
        epoch_task_loss = 0.0 
        epoch_feat_loss = 0.0 
        epoch_resp_loss = 0.0 
        epoch_si_sdr = 0.0 
        t0 = time.time() 
 
        for noisy_wave, clean_wave in loader: 
            noisy_wave = noisy_wave.to(device) 
            clean_wave = clean_wave.to(device) 
 
            opt.zero_grad() 
 
            noisy_grid = frame_signal(noisy_wave, FRAME_SIZE, FRAME_SHIFT).unsqueeze(1)   # (B,1,100,320) 
            clean_grid = frame_signal(clean_wave, FRAME_SIZE, FRAME_SHIFT).unsqueeze(1)   # (B,1,100,320) 
 
            with torch.no_grad(): 
                teacher_out_grid = teacher(noisy_grid)          # (B,1,100,320) -- also populates teacher_grabber.feat 
 
            student_out_grid = student(noisy_grid)              # (B,1,100,320) -- also populates student_grabber.feat 
 
            task_loss = combined_loss(student_out_grid, clean_grid) 
 
            s_feat = student_grabber.feat 
            t_feat = teacher_grabber.feat.detach() 
 
            assert s_feat.shape[2] == t_feat.shape[2], ( 
                f"Bottleneck time-frame count differs: student={s_feat.shape[2]}, " 
                f"teacher={t_feat.shape[2]}." 
            ) 
 
            if s_feat.shape[1] != t_feat.shape[1]: 
                if projection.proj is None:
                    projection.build_if_needed(
                        s_feat.shape[1],
                        t_feat.shape[1],
                        device
                    )

                    # Add the newly created projection parameters
                    # to the optimizer so they are trainable.
                    opt.add_param_group({
                        "params": projection.parameters()
                    })

                s_feat_for_skd = projection(s_feat) 
            else: 
                s_feat_for_skd = s_feat 
 
            feat_loss = skd_loss_fn(s_feat_for_skd, t_feat) 
 
            # --- reconstruct both outputs to real waveforms for the response-based term / SI-SDR --- 
            student_waveform = overlap_add(student_out_grid.squeeze(1))          # (B, 16000) 
            teacher_waveform = overlap_add(teacher_out_grid.squeeze(1)).detach()  # (B, 16000) 
 
            if RESPONSE_KD_WEIGHT > 0.0: 
                sc_loss, mag_loss = mrstft(student_waveform, teacher_waveform) 
                resp_loss = sc_loss + mag_loss 
            else: 
                resp_loss = torch.tensor(0.0, device=device) 
 
            total_loss = ( 
                task_loss 
                + FEATURE_KD_WEIGHT * feat_loss 
                + RESPONSE_KD_WEIGHT * resp_loss 
            ) 
            total_loss.backward() 
            opt.step() 
 
            epoch_task_loss += task_loss.item() 
            epoch_feat_loss += feat_loss.item() 
            epoch_resp_loss += resp_loss.item() if isinstance(resp_loss, torch.Tensor) else resp_loss 
            with torch.no_grad(): 
                epoch_si_sdr += (-si_sdr_loss(student_waveform, clean_wave)).item() 
 
        scheduler.step() 
        n_batches = len(loader) 
        print( 
            f"Epoch {epoch+1}/{EPOCHS} | " 
            f"task_loss: {epoch_task_loss/n_batches:.6f} | " 
            f"feat_kd_loss: {epoch_feat_loss/n_batches:.6f} | " 
            f"resp_kd_loss: {epoch_resp_loss/n_batches:.6f} | " 
            f"SI-SDR: {epoch_si_sdr/n_batches:.3f} dB | " 
            f"LR: {scheduler.get_last_lr()[0]:.6f} | " 
            f"time: {time.time()-t0:.1f}s" 
        ) 
 
        if (epoch + 1) % 10 == 0: 
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"compressed_inceptcn_epoch{epoch+1}.pt") 
            torch.save(student.state_dict(), ckpt_path) 
            print(f"  saved checkpoint -> {ckpt_path}") 
 
    final_path = os.path.join(CHECKPOINT_DIR, "compressed_inceptcn_final.pt") 
    torch.save(student.state_dict(), final_path) 
    print(f"Stage 2 complete. Final compressed-IncepTCN checkpoint: {final_path}") 
 
    teacher_grabber.remove() 
    student_grabber.remove() 
 
 
if __name__ == "__main__": 
    train()