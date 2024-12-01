from typing import Tuple
import numpy as np
import cv2

class PNP:
    def __init__(self, camera_config):
        self.camera_config = camera_config
        self.use_opencv = False
    
    def estimate_pose(self, object_points: np.ndarray, image_points: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        '''
        Estimate pose of object from object points and image points

        Inputs:
            object_points (N x 3 np.ndarray): 3D points of object
            image_points (N x 2 np.ndarray): 2D points of object in image

        Outputs:
            success (bool): True if pose is estimated successfully
            rvec (3 x 1 np.ndarray): Rotation vector in Rodrigues form
            tvec (3 x 1 np.ndarray): Translation vector
        '''
        raise NotImplementedError
    
    def estimate_pose_ransac(self, object_points: np.ndarray, image_points: np.ndarray, max_iterations: int = 100, reprojection_error: float = 8.0) -> Tuple[bool, np.ndarray, np.ndarray]:
        '''
        Estimate pose of object from object points and image points using RANSAC

        Inputs:
            object_points (N x 3 np.ndarray): 3D points of object
            image_points (N x 2 np.ndarray): 2D points of object in image
            max_iterations (int): Maximum number of iterations for RANSAC
            reprojection_error (float): Maximum reprojection error for RANSAC

        Outputs:
            success (bool): True if pose is estimated successfully
            rvec (3 x 1 np.ndarray): Rotation vector in Rodrigues form
            tvec (3 x 1 np.ndarray): Translation vector
        '''
        # TODO IMPLEMENT
        # TODO reprojection error is euclidean distance between projected points and image points