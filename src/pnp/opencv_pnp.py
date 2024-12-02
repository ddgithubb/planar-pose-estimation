from pnp import PNP
import cv2

class OpenCVPNP(PNP):
    def __init__(self, camera_config, pnp_flag=cv2.SOLVEPNP_ITERATIVE):
        super().__init__(camera_config)
        self.use_opencv = True
        self.pnp_flag = pnp_flag

    def estimate_pose(self, object_points, image_points):
        success, rvec, tvec, _ = cv2.solvePnP(object_points, image_points, self.camera_config.camera_matrix, None, flags=self.pnp_flag)
        return success, rvec, tvec