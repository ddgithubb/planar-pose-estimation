from src.camera_config.camera_config import CameraConfig
from src.feature_matcher.feature_matcher import FeatureMatcher
from src.pnp.pnp import PNP
from PIL import Image
import numpy as np
from typing import Tuple
import cv2
import src.graphics as graphics

class PoseEstimator:
    def __init__(self, camera_config: CameraConfig, feature_matcher: FeatureMatcher, pnp: PNP):
        self.camera_config = camera_config
        self.feature_matcher = feature_matcher
        self.pnp = pnp

    def estimate_pose(self, base_image: Image.Image, target_image: Image.Image) -> Tuple[bool, np.ndarray, np.ndarray]:
        object_points, image_points = self.feature_matcher.match(base_image, target_image)
        success, rvec, tvec = self.pnp.estimate_pose(object_points, image_points)
        return success, rvec, tvec
    
    def estimate_pose_ransac(self, base_image: Image.Image, target_image: Image.Image, max_iterations: int = 100, reprojection_error: float = 8.0) -> Tuple[bool, np.ndarray, np.ndarray]:
        object_points, image_points = self.feature_matcher.match(base_image, target_image)
        success, rvec, tvec = self.pnp.estimate_pose_ransac(object_points, image_points, max_iterations, reprojection_error)
        return success, rvec, tvec
    
    def video_estimate_pose(self, base_image_path: str, video_path: str, use_ransac: bool = False, show_frame_pose: bool = True, write_output: bool = True, out_folder: str = 'out') -> None:
        base_img = Image.open(base_image_path)
        height = base_img.size[0]
        width = base_img.size[1]

        base_camera_matrix = self.camera_config.calc_camera_matrix(height, width)
        
        video = cv2.VideoCapture(video_path)
        _, frame = video.read()
        if write_output:
            video_name = video_path.split('/')[-1].split('.')[0]
            video_out = cv2.VideoWriter(f'{out_folder}/{video_name}_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (frame.shape[1], frame.shape[0]))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            height = img.size[0]
            width = img.size[1]
            camera_matrix = self.camera_config.calc_camera_matrix(height, width)

            matched_keypoints1, matched_keypoints2 = self.feature_matcher.match(base_img, img)

            object_points = np.hstack((matched_keypoints1, np.zeros((matched_keypoints1.shape[0], 1))))

            # use camera matrix to convert 2D points to 3D points
            object_points = np.dot(np.linalg.inv(base_camera_matrix), object_points.T).T

            # center points
            object_points = object_points - np.mean(object_points, axis=0)

            image_points = matched_keypoints2
            
            success = False
            rvec = None
            tvec = None
            try:
                if use_ransac:
                    success, rvec, tvec = self.pnp.estimate_pose_ransac(camera_matrix, object_points, image_points)
                else:
                    success, rvec, tvec = self.pnp.estimate_pose(camera_matrix, object_points, image_points)
            except Exception as e:
                print("Exception:", e)
                success = False

            print(f"Success: {success}, rvec: {rvec}, tvec: {tvec}")

            img = frame

            # plot projected points on to image
            if not success:
                print("Failed to solve PnP")
            else:
                if show_frame_pose:
                    axis = np.float32([[0,0,0], [0,0,-1], [0,1,0], [1,0,0]]).reshape(-1,3)
                    axis = 0.1 * axis
                    axis_points, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, None)
                    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)

                    graphics.plot_points_to_image(img, image_points, projected_points, axis_points)
                    graphics.show_image(img)

            if write_output:
                video_out.write(img)

            if cv2.waitKey(1) == 27:
                break

        video.release()
        if write_output:
            video_out.release()
        cv2.destroyAllWindows()
        