#!/usr/bin/env python3

import os
import torch
from torchvision import utils

from datasets.mnist_manager import MNISTDatasetManager, NULL_CLASS
from schedules.beta_schedules import DiffusionSchedule
from sampling.ddim import DDIM
from training.losses import eps_mse_loss
from models import UNetV3


# ---------------- Constants ----------------
LABEL_DROP_PROB = 0.1
GUIDANCE_SCALE = 3.0
DEFAULT_SAMPLE_LABELS = list(range(10)) * 2


def to01(x): return (x.clamp(-1, 1) + 1) / 2


# ---------------- Training script ----------------
def main():

    cond_label_list = DEFAULT_SAMPLE_LABELS

    # ----- Device -----
    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ----- Dataset manager -----
    data = MNISTDatasetManager(device=device)

    # ----- Diffusion process -----
    schedule = DiffusionSchedule(T=1000, schedule="cosine", device=device)
    diffusion = DDIM(schedule, null_label=NULL_CLASS)

    # ----- Model & Optimizer -----
    model = UNetV3(schedule.T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ----- Training params -----
    total_steps = 50000
    batch_size = 64
    log_interval = 500
    sample_interval = 2500
    sample_shape = (20, 1, 28, 28)

    outdir = "output"
    os.makedirs(outdir, exist_ok=True)

    print(f"Training for {total_steps} steps on {device}...")

    running = 0

    for step in range(1, total_steps + 1):

        # ---- Train step ----
        x, labels = data.train_batch(batch_size, drop_prob=LABEL_DROP_PROB)
        loss = eps_mse_loss(model, x, labels, diffusion)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.item()

        # ---- Logging ----
        if step % log_interval == 0:
            model.eval()
            val_x, val_labels = data.val_batch(batch_size * 8)
            val_loss = eps_mse_loss(model, val_x, val_labels, diffusion).item()
            print(f"[{step:05d}] train={running/log_interval:.4f}  val={val_loss:.4f}")
            running = 0

        # ---- Sampling ----
        if step % sample_interval == 0:
            model.eval()

            # unconditional
            imgs = diffusion.sample(model, sample_shape).cpu()
            utils.save_image(to01(imgs), f"{outdir}/samples_step{step}.png", nrow=5)

            # conditional
            cond_labels = data.tiled_labels(cond_label_list, sample_shape[0])
            if cond_labels is not None:
                imgs = diffusion.sample(
                    model, sample_shape, labels=cond_labels, guidance_scale=GUIDANCE_SCALE
                ).cpu()
                utils.save_image(
                    to01(imgs), f"{outdir}/samples_step{step}_class.png", nrow=5
                )

            torch.save(model.state_dict(), f"{outdir}/model_step{step}.pt")

    print("Training complete.")


if __name__ == "__main__":
    main()
