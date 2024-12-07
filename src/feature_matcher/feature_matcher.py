import numpy as np
from typing import Tuple
from PIL import Image

class FeatureMatcher:
    def __init__(self):
        pass

    def match(self, img1: Image.Image, img2: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Given two images, return the matched keypoints in both images

        Inputs:
            img1 (Image.Image): First image
            img2 (Image.Image): Second image
        
        Outputs:
            kpts1 (N x 2 np.ndarray): Keypoints in first image
            kpts2 (N x 2 np.ndarray): Keypoints in second image
            matched_kpts1 (N x 2 np.ndarray): Matched keypoints in first image
            matched_kpts2 (N x 2 np.ndarray): Matched keypoints in second image
        '''
        raise NotImplementedError