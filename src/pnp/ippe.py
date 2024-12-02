from src.pnp.pnp import PNP
import cv2

class IPPE(PNP):
    def __init__(self):
        super().__init__()

    def estimate_pose(self, camera_matrix, object_points, image_points):
        raise NotImplementedError