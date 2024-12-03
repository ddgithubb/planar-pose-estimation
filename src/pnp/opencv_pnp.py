from typing import Tuple
from numpy import ndarray
from src.pnp.pnp import PNP
import cv2

class OpenCVPNP(PNP):
    def __init__(self, pnp_flag=cv2.SOLVEPNP_ITERATIVE):
        super().__init__()
        self.pnp_flag = pnp_flag

    def estimate_pose(self, camera_matrix, object_points, image_points):
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None, flags=self.pnp_flag)
        return success, rvec, tvec
    
    def estimate_pose_ransac(self, camera_matrix: ndarray, object_points: ndarray, image_points: ndarray, max_iterations: int = 100, reprojection_error: float = 8) -> Tuple[bool, ndarray, ndarray]:
        success, rvec, tvec, _ = cv2.solvePnPRansac(object_points, image_points, camera_matrix, None, flags=self.pnp_flag, iterationsCount=max_iterations, reprojectionError=reprojection_error)
        return success, rvec, tvec