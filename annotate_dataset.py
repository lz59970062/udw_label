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
        "You are a visual effects expert. Compare a clear underwater image (Image A) and its degraded version (Image B).\n\n"
        "Provide two descriptive 'instructions' or 'narratives' using plain, sensory language. Avoid technical jargon like 'Dark Channel Prior', 'histogram', or 'Wiener filter'. "
        "Do NOT use terms 'Image A' or 'Image B' in the final descriptions; refer to the actual content (e.g., 'the sea turtle', 'the corals', 'the water').\n\n"
        "1. 'degradation': Describe the 'methods' to degrade the clear scene so it matches the degraded one. "
        "Example: 'Wrap the entire scene in a heavy, greenish fog that obscures the distant reef. Blur the fine textures on the turtle's shell and significantly dim the overall lighting to make it look murky.'\n\n"
        "2. 'enhancement': Describe the 'methods' to restore the degraded scene to its original clarity. "
        "Example: 'Strip away the thick cyan color cast to reveal the natural colors of the coral. Sharpen the blurred edges of the sea turtle and clear up the hazy water to bring out the hidden details in the background.'\n\n"
        "Requirements:\n"
        "- The tone should be descriptive and action-oriented (what needs to change/what happened).\n"
        "- Each description must be a self-contained, stand-alone observation. Strictly avoid any comparative language like 'than the other image', 'more than the reference', or mentioning that a comparison is taking place.\n"
        "- Focus on visual quality: color, clarity, contrast, and detail.\n"
        "- Format as a JSON object with keys 'degradation' and 'enhancement'."
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
        return {"degradation": None, "enhancement": None}

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
            # 跳过已经处理过的
            if 'enhancement' in bad_entry and bad_entry['enhancement']:
                continue
                
            bad_path = os.path.join(IMAGE_ROOT, bad_entry['file'])
            print(f"[{i+1}/{total_pairs}] Processing pair: {pair['good']} <-> {bad_entry['file']}")
            
            res = get_descriptions(good_path, bad_path)
            
            # 新增字段保存
            bad_entry['degradation'] = res.get('degradation')
            bad_entry['enhancement'] = res.get('enhancement')

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
