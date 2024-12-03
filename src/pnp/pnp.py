from typing import Tuple
import numpy as np
import cv2

class PNP:
    def __init__(self):
        pass

    def estimate_pose(self, camera_matrix: np.ndarray, object_points: np.ndarray, image_points: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        '''
        Estimate pose of object from object points and image points

        Inputs:
            camera_matrix (3 x 3 np.ndarray): Camera matrix
            object_points (N x 3 np.ndarray): 3D points of object
            image_points (N x 2 np.ndarray): 2D points of object in image

        Outputs:
            success (bool): True if pose is estimated successfully
            rvec (3 x 1 np.ndarray): Rotation vector in Rodrigues form
            tvec (3 x 1 np.ndarray): Translation vector
        '''
        raise NotImplementedError
    
    def estimate_pose_ransac(self, camera_matrix: np.ndarray, object_points: np.ndarray, image_points: np.ndarray, max_iterations: int = 100, reprojection_error: float = 8.0) -> Tuple[bool, np.ndarray, np.ndarray]:
        '''
        Estimate pose of object from object points and image points using RANSAC

        Inputs:
            camera_matrix (3 x 3 np.ndarray): Camera matrix
            object_points (N x 3 np.ndarray): 3D points of object
            image_points (N x 2 np.ndarray): 2D points of object in image
            max_iterations (int): Maximum number of iterations for RANSAC
            reprojection_error (float): Maximum reprojection error for RANSAC

        Outputs:
            success (bool): True if pose is estimated successfully
            rvec (3 x 1 np.ndarray): Rotation vector in Rodrigues form
            tvec (3 x 1 np.ndarray): Translation vector
        '''