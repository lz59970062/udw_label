import sys
import os
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from qflux.data.dataset import ImageDataset
from qflux.data.config import DatasetInitArgs

# Mimic the config loading
config_path = "configs/udw_finetune.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_config = config["data"]["init_args"]
processor_config = data_config["processor"]

# Create a mock object for data_config since ImageDataset expects a DatasetInitArgs or similar
# The code in dataset.py uses data_config.dataset_path, etc.
# But DatasetInitArgs is a Pydantic model or similar dataclass? 
# Let's see how it's used. In __init__, it does:
# self.data_config = data_config
# dataset_path = data_config.dataset_path
# So it expects an object with attributes, not a dict.

class MockConfig:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if k == "processor":
                setattr(self, k, MockConfig(v))
            else:
                setattr(self, k, v)
        # Default defaults
        if not hasattr(self, "use_cache"): self.use_cache = False
        if not hasattr(self, "cache_dir"): self.cache_dir = None
        if not hasattr(self, "selected_control_indexes"): self.selected_control_indexes = None
        if not hasattr(self, "prompt_empty_drop_keys"): self.prompt_empty_drop_keys = []
        if not hasattr(self, "caption_dropout_rate"): self.caption_dropout_rate = 0.0

dataset_config_obj = MockConfig(data_config)

print("Initializing ImageDataset...")
ds = ImageDataset(dataset_config_obj)
print(f"Dataset length: {len(ds)}")

if len(ds) > 0:
    print("Loading first sample...")
    sample = ds[0]
    print("Sample keys:", sample.keys())
    print("Image shape:", sample["image"].shape)
    print("Control shape:", sample["control"].shape)
    print("Prompt:", sample["prompt"])
else:
    print("Dataset is empty!")
