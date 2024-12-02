from pnp.pnp import PNP
import cv2

class IPPE(PNP):
    def __init__(self, camera_config):
        super().__init__(camera_config)

    def estimate_pose(self, object_points, image_points):
        raise NotImplementedError