import sys
import os
import yaml
import unittest.mock as mock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mock dependencies that trigger the diffusers import error or are not needed for simple loading test
sys.modules["qflux.data.cache_manager"] = mock.MagicMock()
sys.modules["qflux.losses.edit_mask_loss"] = mock.MagicMock()
sys.modules["qflux.utils.huggingface"] = mock.MagicMock()
# qflux.utils.tools is needed for hash_string_md5 potentially, but we can mock it if it imports heavy stuff.
# The traceback showed qflux.utils.__init__ imports get_model_config -> models -> crash.
# So we must prevent qflux.utils from chaining imports.
sys.modules["qflux.utils"] = mock.MagicMock() 
sys.modules["qflux.utils.tools"] = mock.MagicMock()
# We need to expose instantiate_class though if it's used in __init__
# dataset.py: from qflux.utils.tools import hash_string_md5, pad_to_max_shape, instantiate_class (inside method)

# Let's just mock the imports inside dataset.py by patching sys.modules BEFORE importing dataset
# But dataset.py imports them at top level.

# Re-implement minimal tools needed
def mock_hash(x): return "hash"
def mock_pad(x): return x

mock_tools = mock.MagicMock()
mock_tools.hash_string_md5 = mock_hash
mock_tools.pad_to_max_shape = mock_pad
sys.modules["qflux.utils.tools"] = mock_tools

# Now import
from qflux.data.dataset import ImageDataset

# Config setup
config_path = "configs/udw_finetune.yaml"
# We need to read the config and fix the path to be absolute or correct relative to current dir
# because the test is running in qwen-image-finetune/
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_config = config["data"]["init_args"]
# Point to the actual file location relative to where we are or absolute
# The user's file is at /home/lizhi_2024/program/udw_dataset/dataset_labeled.yaml
# We are likely running from /home/lizhi_2024/program/udw_dataset/qwen-image-finetune
# So path should be OK if absolute.

class MockConfig:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if k == "processor" and isinstance(v, dict):
                setattr(self, k, MockConfig(v))
            else:
                setattr(self, k, v)
        if not hasattr(self, "use_cache"): self.use_cache = False
        if not hasattr(self, "cache_dir"): self.cache_dir = None
        if not hasattr(self, "selected_control_indexes"): self.selected_control_indexes = None
        if not hasattr(self, "prompt_empty_drop_keys"): self.prompt_empty_drop_keys = []
        if not hasattr(self, "caption_dropout_rate"): self.caption_dropout_rate = 0.0

dataset_config_obj = MockConfig(data_config)

# Mock instantiation of processor
with mock.patch("qflux.utils.tools.instantiate_class") as mock_instantiate:
    mock_processor = mock.MagicMock()
    mock_processor.preprocess.return_value = {"image": "processed"} # Minimal return
    mock_instantiate.return_value = mock_processor
    
    # We also need to patch _load_huggingface_dataset etc to avoid issues, though we are using local_yaml
    
    print("Initializing ImageDataset (with mocks)...")
    try:
        ds = ImageDataset(dataset_config_obj)
        print(f"Dataset length: {len(ds)}")

        if len(ds) > 0:
            print("Loading first sample to check parsing...")
            # We access ds.all_samples directly to check raw data because __getitem__ does preprocessing/caching which we mocked aggressively
            sample = ds.all_samples[0]
            print("Sample data:", sample)
            
            # verify paths exist
            if os.path.exists(sample["image"]):
                print("Image path exists.")
            else:
                print(f"Image path NOT found: {sample['image']}")
                
            if os.path.exists(sample["control"][0]):
                print("Control path exists.")
            else:
                print(f"Control path NOT found: {sample['control'][0]}")

            print("Prompt:", sample["caption"])
        else:
            print("Dataset is empty!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
