import numpy as np
from typing import Tuple
from src.feature_matcher.feature_matcher import FeatureMatcher
import torch
from PIL import Image
import torchvision.transforms as tfm

# import matching from image_matching_models package
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../image-matching-models"))

from matching import get_matcher

default_device = "cpu"
if torch.cuda.is_available():
    default_device = "cuda"

print(f"Using default device: {default_device}")

class ImageMatchingModels(FeatureMatcher):
    def __init__(self, matcher_name="sift-nn", device=default_device):
        super().__init__()
        self.matcher = get_matcher(matcher_name, device=device)
        self.device = device

    def match_tensor(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        result = self.matcher(img1, img2)
        return result["matched_kpts0"], result["matched_kpts1"]
        
    def match(self, img1: Image.Image, img2: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        img1_tensor = tfm.ToTensor()(img1)
        img2_tensor = tfm.ToTensor()(img2)
        return self.match_tensor(img1_tensor, img2_tensor)