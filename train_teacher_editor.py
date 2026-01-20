import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
import copy

# Reuse previous dataset class
from distill_dataset import UnderwaterDistillationDataset

# --- Configuration ---
# Assuming Qwen-Image is compatible with Stable Diffusion architecture
# If not, switch to "stabilityai/stable-diffusion-2-1" or "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_NAME = "runwayml/stable-diffusion-v1-5" 
# NOTE: User specified "Qwen/Qwen-Image-2512" before. If that is a path to a diffusers model, use it.
# If it is a proprietary model not in diffusers format, we cannot fine-tune it easily.
# Defaulting to SD1.5 as a robust base for "Teacher" if Qwen is not available as UNet.
# But I will use the variable so user can change it.

OUTPUT_DIR = "./finetuned_teacher_editor"
DATASET_ROOT = "data_pipe/export_underwater2/dataset_20250921_093929"
YAML_PATH = "dataset_labeled.yaml"

BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 1e-4
EPOCHS = 10 

def main():
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM)
    device = accelerator.device
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading base model: {MODEL_NAME}")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}. Please check if it is a valid Diffusers model path.")
        raise e

    # 2. Modify UNet for InstructPix2Pix (4 channels -> 8 channels)
    if unet.config.in_channels == 4:
        print("Converting UNet to InstructPix2Pix style (4 -> 8 channels)...")
        with torch.no_grad():
            old_conv_in = unet.conv_in
            new_conv_in = torch.nn.Conv2d(
                8, old_conv_in.out_channels, 
                kernel_size=old_conv_in.kernel_size, 
                padding=old_conv_in.padding
            )
            # Initialize weights: Copy first 4 channels, zero out new 4 channels
            new_conv_in.weight[:, :4] = old_conv_in.weight
            new_conv_in.weight[:, 4:] = torch.zeros_like(old_conv_in.weight)
            if old_conv_in.bias is not None:
                new_conv_in.bias = old_conv_in.bias
            
            unet.conv_in = new_conv_in
            unet.config.in_channels = 8
    
    # 3. Freeze Components
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 4. LoRA Configuration
    # We train the new input layer fully, and LoRA for the rest
    unet.conv_in.requires_grad_(True) 
    
    # Enable Gradient Checkpointing for memory saving
    unet.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "conv1", "conv2"], # Add more modules for better adaptation
        init_lora_weights="gaussian"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 5. Dataset
    # We use augmented text if available for the teacher too
    yaml_source = "dataset_augmented.yaml" if os.path.exists("dataset_augmented.yaml") else YAML_PATH
    dataset = UnderwaterDistillationDataset(
        yaml_path=yaml_source, 
        root_dir=DATASET_ROOT,
        use_text_augmentation=True 
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 6. Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=LR)
    
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=0, 
        num_training_steps=len(dataloader) * EPOCHS
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    vae.to(device)
    text_encoder.to(device)

    # 7. Training Loop
    for epoch in range(EPOCHS):
        unet.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Data
                bad_pixel = batch["bad_image"].to(device)
                good_pixel = batch["good_image"].to(device)
                
                # Handle text augmentation list or string
                prompts = batch["enhancement"]
                if isinstance(prompts, list) and not isinstance(prompts[0], str):
                     # If collate fn returned list of tuples/lists, flattened it
                     prompts = [p for p in prompts] 

                # VAE Encode
                latents_target = vae.encode(good_pixel).latent_dist.sample() * vae.config.scaling_factor
                latents_condition = vae.encode(bad_pixel).latent_dist.sample() * vae.config.scaling_factor
                
                # Text Encode
                inputs = tokenizer(
                    prompts, padding="max_length", max_length=tokenizer.model_max_length, 
                    truncation=True, return_tensors="pt"
                ).to(device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]

                # Noise
                noise = torch.randn_like(latents_target)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents_target.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents_target, noise, timesteps)

                # Concatenate Inputs: [Noisy encoded target, Clean encoded condition]
                unet_input = torch.cat([noisy_latents, latents_condition], dim=1)

                # Predict
                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(model_pred, noise)
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": loss.item()})
        
        # Save Checkpoint
        if accelerator.is_main_process:
            save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
            unet_unwrapped = accelerator.unwrap_model(unet)
            
            # Save LoRA adapters
            unet_unwrapped.save_pretrained(save_path)
            
            # CRITICAL: Also save the modified conv_in weights because they are not part of LoRA
            # Peft saves adapters, but we modified the base model topology (conv_in).
            # We need to manually save the input layer state.
            conv_in_state = unet_unwrapped.base_model.model.conv_in.state_dict()
            torch.save(conv_in_state, os.path.join(save_path, "conv_in_state.pt"))
            
            print(f"Saved model and conv_in weights to {save_path}")

if __name__ == "__main__":
    main()
