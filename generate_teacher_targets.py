import os
import yaml
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import DiffusionPipeline

# --- Config ---
YAML_PATH = "dataset_labeled.yaml"
OUTPUT_DIR = "images_teacher"
MODEL_NAME = "Qwen/Qwen-Image-2512" 
dataset_root = "data_pipe/export_underwater2/dataset_20250921_093929" 
def generate_teacher_data():
    if not os.path.exists(YAML_PATH):
        print(f"Error: {YAML_PATH} not found.")
        return

    # 1. Load Dataset Config
    with open(YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Load Teacher Model
    print(f"Loading Teacher Model: {MODEL_NAME}...")
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"

    try:
        pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have access to the model and enough VRAM.")
        return

    negative_prompt = "low resolution, low quality, distortion, artifacts, ugly, dim, dark, noise, blurry text"
    
    # 3. Inference Loop
    pairs = data.get('pairs', [])
    print(f"Starting generation for {len(pairs)} pairs...")
    
    count = 0
    for pair in tqdm(pairs):
        if not pair.get('bads'): continue
        
        bad_entry = pair['bads'][0]
        bad_rel_path = bad_entry['file']
        bad_full_path = os.path.join(dataset_root, bad_rel_path) # Changed to use dataset_root
        prompt = bad_entry['enhancement'] 
        
        filename = os.path.basename(bad_rel_path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(save_path):
            continue
            
        try:
            # We use the 'enhancement' prompt to generate the ideal image
            # Since Qwen-Image is T2I or I2I, here we treat it as T2I guided by the prompt 
            # (or I2I if we supported loading the init image, but straightforward generation is safer for diverse prompts)
            
            # Optional: Read bad image for aspect ratio
            if os.path.exists(bad_full_path):
                img = Image.open(bad_full_path)
                w, h = img.size
                # Round to nearest 32
                w = (w // 32) * 32
                h = (h // 32) * 32
                w = min(w, 1024)
                h = min(h, 1024)
            else:
                w, h = 1024, 1024

            generated_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=w, 
                height=h,
                num_inference_steps=30,
                guidance_scale=7.0
            ).images[0]
            
            generated_image.save(save_path)
            count += 1
            
        except Exception as e:
            print(f"Failed {filename}: {e}")
            torch.cuda.empty_cache()

    print(f"Generation complete. Generated {count} new images in {OUTPUT_DIR}/")

if __name__ == "__main__":
    generate_teacher_data()
#TODO 先微调再进行教师图像生成 