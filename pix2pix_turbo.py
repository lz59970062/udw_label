import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from peft import LoraConfig

# --- VAE Hacks for Pix2Pix-Turbo (Skip Connections) ---
def my_vae_encoder_fwd(self, sample):
    # This stores activations to be used as skip connections in the decoder
    self.current_down_blocks = []
    
    sample = self.conv_in(sample)
    for down_block in self.down_blocks:
        self.current_down_blocks.append(sample)
        sample = down_block(sample)
    
    sample = self.mid_block(sample)
    sample = self.conv_out(sample)
    sample = self.conv_norm_out(sample)
    sample = F.silu(sample)
    return sample

def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    sample = self.mid_block(sample, latent_embeds)
    
    # Apply skip connections from the encoder
    # Note: indices depend on the number of blocks. SD-Turbo VAE typically has 4.
    skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
    
    for i, up_block in enumerate(self.up_blocks):
        if not self.ignore_skip and i < len(self.incoming_skip_acts):
            # Reverse order of skip activations
            skip_act = self.incoming_skip_acts[-(i + 1)]
            # Projection to match channels if necessary
            skip_act = skip_convs[i](skip_act)
            sample = sample + skip_act * self.gamma
        sample = up_block(sample, latent_embeds)
        
    sample = self.conv_norm_out(sample)
    sample = F.silu(sample)
    sample = self.conv_out(sample)
    return sample

class ConvGRUModulator(nn.Module):
    def __init__(self, latent_channels, qmap_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = latent_channels
        self.qmap_proj = nn.Conv2d(qmap_channels, latent_channels, kernel_size=1)
        self.conv_gates = nn.Conv2d(latent_channels + latent_channels, 2 * hidden_channels, kernel_size=3, padding=1)
        self.conv_candidate = nn.Conv2d(latent_channels + latent_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, latent, qmap, n_steps=1):
        if qmap.shape[2:] != latent.shape[2:]:
            qmap_aligned = F.interpolate(qmap, size=latent.shape[2:], mode='bilinear', align_corners=False)
        else:
            qmap_aligned = qmap
        
        qmap_feat = self.qmap_proj(qmap_aligned)
        hidden_state = latent

        for _ in range(n_steps):
            combined_input = torch.cat([qmap_feat, hidden_state], dim=1)
            gates = self.conv_gates(combined_input)
            reset_gate, update_gate = gates.chunk(2, 1)
            reset_gate, update_gate = torch.sigmoid(reset_gate), torch.sigmoid(update_gate)
            candidate_input = torch.cat([qmap_feat, reset_gate * hidden_state], dim=1)
            candidate_hidden = torch.tanh(self.conv_candidate(candidate_input))
            hidden_state = (1 - update_gate) * hidden_state + update_gate * candidate_hidden
        return hidden_state

class PatchDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator similar to CycleGAN/Pix2Pix.
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, model_id="stabilityai/sd-turbo", lora_rank_unet=8, lora_rank_vae=4, use_qmap=False, qmap_steps=1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        self.sched = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.sched.set_timesteps(1) # For 1-step generation
        
        self.qmap_steps = qmap_steps
        self.use_qmap = use_qmap

        # Inject VAE hacks
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(self.vae.encoder, self.vae.encoder.__class__)
        self.vae.decoder.forward = my_vae_decoder_fwd.__get__(self.vae.decoder, self.vae.decoder.__class__)
        
        # Add skip connection convs helper
        self.vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, (1,1), bias=False)
        self.vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, (1,1), bias=False)
        self.vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, (1,1), bias=False)
        self.vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, (1,1), bias=False)
        self.vae.decoder.ignore_skip = False
        self.vae.decoder.gamma = 1.0

        # Initialize LoRA
        target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
        vae_lora_config = LoraConfig(r=lora_rank_vae, target_modules=target_modules_vae)
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

        target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out", "proj_in", "proj_out"]
        unet_lora_config = LoraConfig(r=lora_rank_unet, target_modules=target_modules_unet)
        self.unet.add_adapter(unet_lora_config)

        if use_qmap:
            self.conv_gru_modulator = ConvGRUModulator(self.vae.config.latent_channels, 32)

        self.timesteps = torch.tensor([999])

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, p in self.unet.named_parameters():
            if "lora" in n: p.requires_grad = True
        for n, p in self.vae.named_parameters():
            if "lora" in n or "skip" in n: p.requires_grad = True

    def forward(self, c_t, prompt, qmap=None):
        caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(c_t.device)
        caption_enc = self.text_encoder(caption_tokens)[0]
        
        # Step 1: Encode image to latent
        # Note: self.vae.encoder.current_down_blocks is populated here
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        
        unet_input = encoded_control
        if qmap is not None and self.use_qmap:
            unet_input = self.conv_gru_modulator(encoded_control, qmap, n_steps=self.qmap_steps)

        # Step 2: 1-step UNet prediction
        model_pred = self.unet(unet_input, self.timesteps.to(c_t.device), encoder_hidden_states=caption_enc).sample
        
        # Step 3: Denoising step
        x_denoised = self.sched.step(model_pred, self.timesteps[0], unet_input).prev_sample
        
        # Step 4: Decode with skip connections
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        
        return output_image
