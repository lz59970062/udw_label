import yaml
import os
import base64
from openai import OpenAI
from pathlib import Path
import json

# 配置
API_KEY = "EMPTY"
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
YAML_PATH = "./data_pipe/export_underwater2/dataset_20250921_093929/dataset.yaml"
IMAGE_ROOT = os.path.dirname(YAML_PATH)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_descriptions(good_img_path, bad_img_path):
    good_b64 = encode_image(good_img_path)
    bad_b64 = encode_image(bad_img_path)
    
    prompt = (
        "You are an elite expert in underwater optics, computer vision, and artistic photography analysis. "
        "I am providing two images for a comparative study: "
        "Image A is the reference (high quality/clean), and Image B is the degraded version (underwater environment artifacts).\n\n"
        "Please provide an extremely detailed, technical, and descriptive analysis for two scenarios:\n\n"
        "1. 'a2b' (Degradation Process): Describe the physical and perceptual transformation from A to B. "
        "Discuss global factors like hydrodynamic haze, light absorption (loss of red/long-wavelength light), backscattering (marine snow), and contrast attenuation. "
        "Analyze local changes such as the blurring of edge gradients on specific objects, loss of micro-textures, and the introduction of chromatic noise in shadow regions.\n\n"
        "2. 'b2a' (Restoration/Enhancement): Describe the restorative process required to transform B back into A. "
        "Detail the removal of color casts (e.g., 'de-greening' or 'de-blueing'), the recovery of the dark channel, and the reconstruction of high-frequency details. "
        "Mention how structural integrity is regained and how the dynamic range is expanded to reveal hidden details in previously obscured areas.\n\n"
        "Requirements:\n"
        "- Use professional and descriptive language suitable for training Large Multimodal Models (LMMs).\n"
        "- Ensure descriptions are cohesive, focusing on the causal relationship between the two images.\n"
        "- Format your response as a valid JSON object with keys 'a2b' and 'b2a'."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{good_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{bad_b64}"}},
                    ],
                }
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error processing {good_img_path} and {bad_img_path}: {e}")
        return {"a2b": None, "b2a": None}

def main():
    with open(YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)
    YAML_PATH_labeled = YAML_PATH.replace('.yaml', '_labeled.yaml')
    if 'pairs' not in data:
        print("No pairs found in YAML.")
        return

    total_pairs = len(data['pairs'])
    print(f"Starting annotation for {total_pairs} pairs...")

    for i, pair in enumerate(data['pairs']):
        good_path = os.path.join(IMAGE_ROOT, pair['good'])
        
        for bad_entry in pair['bads']:
            # 跳过已经处理过的（可选）
            if 'a2b_desc' in bad_entry and bad_entry['a2b_desc']:
                continue
                
            bad_path = os.path.join(IMAGE_ROOT, bad_entry['file'])
            print(f"[{i+1}/{total_pairs}] Processing pair: {pair['good']} <-> {bad_entry['file']}")
            
            res = get_descriptions(good_path, bad_path)
            
            # 新增字段保存
            bad_entry['a2b_desc'] = res.get('a2b')
            bad_entry['b2a_desc'] = res.get('b2a')

        # 每处理 10 个保存一次，防止长时间运行中断
        if (i + 1) % 10 == 0:
            with open(YAML_PATH_labeled, 'w') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
            print(f"Saved progress at {i+1} pairs.")

    # 最终保存
    with open(YAML_PATH_labeled, 'w') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print("Annotation completed successfully.")

if __name__ == "__main__":
    main()
