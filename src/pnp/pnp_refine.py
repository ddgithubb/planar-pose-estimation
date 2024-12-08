from src.pnp.pnp import PNP

class PNPRefine(PNP):
    def __init__(self, pnp: PNP):
        super().__init__(f"{pnp.name} + Refine")
        self.pnp = pnp

    def estimate_pose(self, camera_matrix, object_points, image_points):
        success, rvec, tvec = self.pnp.estimate_pose(camera_matrix, object_points, image_points)
        if not success:
            return False, None, None
        success, rvec, tvec = self.refine(camera_matrix, object_points, image_points, rvec, tvec)
        return success, rvec, tvec

    def estimate_pose_ransac(self, camera_matrix, object_points, image_points, max_iterations=100, reprojection_error=8.0):
        success, rvec, tvec = self.pnp.estimate_pose_ransac(camera_matrix, object_points, image_points, max_iterations, reprojection_error)
        if not success:
            return False, None, None
        success, rvec, tvec = self.refine(camera_matrix, object_points, image_points, rvec, tvec)
        return success, rvec, tvec
