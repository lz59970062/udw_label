import yaml
import os
import json
from openai import OpenAI
from tqdm import tqdm

# --- Config ---
API_KEY = "EMPTY" 
BASE_URL = "http://localhost:8000/v1" 
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct" 
YAML_PATH = "dataset_labeled.yaml"
OUTPUT_YAML_PATH = "dataset_augmented.yaml"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """You are an expert AI prompt engineer for image restoration models. 
Your task is to augment a given image enhancement instruction into 3 diverse variations while keeping the original meaning intact.

Variations needed:
1. "concise": A short, punchy command (under 15 words).
2. "detailed": A rich, sensory description focusing on light, texture, and clarity (2-3 sentences).
3. "technical": A description that sounds slightly more technical (e.g., "correct white balance", "reduce turbidity", "enhance contrast").

Output must be valid JSON with keys: "concise", "detailed", "technical".
"""

def augment_instruction(original_text):
    if not original_text: return None
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Original Instruction: {original_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=512
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"\nError augmenting: {e}")
        return None

def main():
    if not os.path.exists(YAML_PATH):
        print(f"Config {YAML_PATH} not found.")
        return

    print(f"Loading {YAML_PATH}...")
    with open(YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)
    
    pairs = data.get('pairs', [])
    print(f"Augmenting text for {len(pairs)} pairs...")
    
    processed_count = 0
    
    for pair in tqdm(pairs):
        if not pair.get('bads'): continue
        
        for bad_entry in pair['bads']:
            original_enhance = bad_entry.get('enhancement')
            
            # Skip conditions
            if not original_enhance: continue
            if 'enhancement_variations' in bad_entry: continue 

            variations = augment_instruction(original_enhance)
            
            if variations:
                # Combine original with new variations
                bad_entry['enhancement_variations'] = [
                    original_enhance,
                    variations.get('concise'),
                    variations.get('detailed'),
                    variations.get('technical')
                ]
                # Filter out None or empty strings
                bad_entry['enhancement_variations'] = [v for v in bad_entry['enhancement_variations'] if v]
                
                processed_count += 1
        
        # Save periodically
        if processed_count > 0 and processed_count % 20 == 0:
             with open(OUTPUT_YAML_PATH, 'w') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    # Final save
    with open(OUTPUT_YAML_PATH, 'w') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"Done. Augmented {processed_count} instructions. Saved to {OUTPUT_YAML_PATH}")

if __name__ == "__main__":
    main()
