import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from distill_dataset import UnderwaterDistillationDataset
from accelerate import Accelerator
from lpips import LPIPS

# --- Configuration ---
TEACHER_MODEL = "Qwen/Qwen-Image-2512" # Note: Qwen-Image-2512 is often used via pipeline
# For practical local distillation to a small model like SD-v1.5 or custom small UNet
STUDENT_BASE = "runwayml/stable-diffusion-v1-5" 
OUTPUT_DIR = "./distilled_underwater_model"
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
NUM_EPOCHS = 10
WEIGHT_LPIPS = 0.5
GRADIENT_ACCUMULATION_STEPS = 4

def train():
    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    device = accelerator.device

    # 1. Load Teacher Model (as an emulator or generator if needed)
    # Since Qwen-Image is a full MLLM/Diffusion hybrid, we assume it's used to provide targets
    # or we treat the 'good' images in dataset as teacher's ground truth.
    
    # 2. Load Student Model (SD v1.5 components for fine-tuning)
    tokenizer = CLIPTokenizer.from_pretrained(STUDENT_BASE, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(STUDENT_BASE, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(STUDENT_BASE, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(STUDENT_BASE, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(STUDENT_BASE, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # 3. Loss Functions
    l1_loss = nn.L1Loss()
    lpips_loss = LPIPS(net='alex').to(device) # perceptual loss

    # 4. Dataset & Dataloader
    dataset = UnderwaterDistillationDataset(
        yaml_path="dataset_labeled.yaml",
        root_dir="."
    )
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # 6. Prepare for training
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    vae.to(device)
    text_encoder.to(device)

    # 7. Training Loop
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Process inputs
                bad_imgs = batch["bad_image"].to(device)
                good_imgs = batch["good_image"].to(device)
                prompt = batch["enhancement"] # Use enhancement description as prompt

                # Latent representation
                latents = vae.encode(bad_imgs).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                target_latents = vae.encode(good_imgs).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor

                # Encode text
                inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]

                # Noise prediction (Distillation: Predict the target directly or noise that leads to target)
                # Here we implement a simplified Image-to-Image distillation logic
                # For ControlNet-like or direct mapping:
                
                # Sample noise
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                
                # We want the student to reconstruct the "good" latent from a "bad" latent + prompt
                # Simplification: predict the delta or the target
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                
                # Condition on the "bad" image (similar to InstructPix2Pix)
                # For SD 1.5, we might need to concat latents or use a ControlNet. 
                # This template assumes direct fine-tuning of the UNet on the image-pair task.
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Loss computation
                loss_mse = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Optional: Pixel-level distillation loss
                # decoded_student = vae.decode(model_pred / vae.config.scaling_factor).sample
                # loss_lpips = lpips_loss(decoded_student, good_imgs).mean()
                
                loss = loss_mse # + WEIGHT_LPIPS * loss_lpips

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

    # Save the model
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(OUTPUT_DIR, "unet"))
        print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
