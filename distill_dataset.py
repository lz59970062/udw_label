import yaml
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

import random

class UnderwaterDistillationDataset(Dataset):
    def __init__(self, yaml_path, root_dir, teacher_dir=None, transform=None, use_text_augmentation=False):
        with open(yaml_path, 'r') as f:
            self.data = yaml.safe_load(f)
        self.root_dir = root_dir
        self.teacher_dir = teacher_dir
        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.use_text_aug = use_text_augmentation
        
        # Pre-process pairs to flatten them if necessary, or just use as is
        self.pairs = self.data.get('pairs', [])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        good_rel_path = pair['good']
        
        if not pair.get('bads'):
            return self.__getitem__((idx + 1) % len(self))
            
        bad_entry = pair['bads'][0]
        bad_rel_path = bad_entry['file']
        degradation_desc = bad_entry.get('degradation', "")
        
        # --- Text Augmentation Logic ---
        base_enhancement = bad_entry.get('enhancement', "")
        enhancement_desc = base_enhancement
        
        if self.use_text_aug and 'enhancement_variations' in bad_entry:
            variations = bad_entry['enhancement_variations']
            if variations:
                enhancement_desc = random.choice(variations)
        # -------------------------------

        good_img_path = os.path.join(self.root_dir, good_rel_path)
        bad_img_path = os.path.join(self.root_dir, bad_rel_path)


        try:
            good_img = Image.open(good_img_path).convert('RGB')
            bad_img = Image.open(bad_img_path).convert('RGB')
            
            teacher_img = None
            if self.teacher_dir:
                bad_filename = os.path.basename(bad_rel_path)
                teacher_path = os.path.join(self.teacher_dir, bad_filename)
                if os.path.exists(teacher_path):
                    teacher_img_pil = Image.open(teacher_path).convert('RGB')
                    if self.transform:
                        teacher_img = self.transform(teacher_img_pil)
                else:
                    # Fallback to good image if teacher gen is missing
                    if self.transform:
                        teacher_img = self.transform(good_img)
            
        except Exception as e:
            print(f"Error loading images: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            good_img = self.transform(good_img)
            bad_img = self.transform(bad_img)

        item = {
            "bad_image": bad_img,
            "good_image": good_img,
            "degradation": degradation_desc,
            "enhancement": enhancement_desc
        }
        
        if teacher_img is not None:
            item["teacher_image"] = teacher_img
            
        return item
