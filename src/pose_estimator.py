from src.camera_config.camera_config import CameraConfig
from src.feature_matcher.feature_matcher import FeatureMatcher
from src.pnp.pnp import PNP
from src.analytics import Analytics
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
    
    def keypoints_to_plane(self, camera_matrix, keypoints):
        center_location = np.array([camera_matrix[1, 2], camera_matrix[0, 2], 0])
        object_points = np.hstack((keypoints, np.zeros((keypoints.shape[0], 1))))
        object_points = np.vstack((object_points, center_location))

        # use camera matrix to convert 2D points to 3D points
        object_points = np.dot(np.linalg.inv(camera_matrix), object_points.T).T

        # center points
        # object_points = object_points - np.mean(object_points, axis=0)

        object_points = object_points[:-1] - object_points[-1]

        return object_points

    def video_estimate_pose(
            self, 
            base_image_path: str, 
            video_path: str, 
            out_name: str = None,
            use_ransac: bool = False,
            ransac_max_iterations: int = 1000,
            use_lucas_kanade: bool = False,
            use_refine: bool = False,
            verbose: bool = False,
            show_frame_pose: bool = False, 
            write_output: bool = False, 
            out_folder: str = 'out',
            analytics: Analytics = None,
            continue_on_failure: bool = True
            ) -> None:
        base_img = Image.open(base_image_path)
        height = base_img.size[0]
        width = base_img.size[1]

        base_camera_matrix = self.camera_config.calc_camera_matrix(height, width)
        
        video = cv2.VideoCapture(video_path)
        _, frame = video.read()
        if write_output:
            if out_name is None:
                out_name = video_path.split('/')[-1].split('.')[0]
            video_out = cv2.VideoWriter(f'{out_folder}/{out_name}_output.mp4', cv2.VideoWriter_fourcc(*'X264'), 30.0, (frame.shape[1], frame.shape[0]))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        height = img.size[0]
        width = img.size[1]
        camera_matrix = self.camera_config.calc_camera_matrix(height, width)

        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame = original_frame
        kpts1, _, matched_keypoints1, matched_keypoints2 = self.feature_matcher.match(base_img, img)
        all_object_points = self.keypoints_to_plane(base_camera_matrix, kpts1)
        object_points = self.keypoints_to_plane(base_camera_matrix, matched_keypoints1)
        image_points = matched_keypoints2
        prev_image_points = image_points

        if analytics is not None:
            analytics.start()

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if analytics is not None:
                analytics.start_frame()
                if verbose:
                    print(f"\nFrame: {analytics.num_frames}")


            if analytics is not None:
                analytics.start_feature_match()

            if use_lucas_kanade:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image_points, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, prev_image_points, None)

                image_points = image_points[st.flatten() == 1]
                object_points = object_points[st.flatten() == 1]
                
                prev_frame = gray_frame
                prev_image_points = image_points
            else:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                _, _, matched_keypoints1, matched_keypoints2 = self.feature_matcher.match(base_img, img)

                object_points = self.keypoints_to_plane(base_camera_matrix, matched_keypoints1)
                image_points = matched_keypoints2

            if analytics is not None:
                analytics.end_feature_match()
            
            success = False
            rvec = None
            tvec = None
            
            if analytics is not None:
                analytics.start_pnp()

            if continue_on_failure:
                try:
                    if use_ransac:
                        success, rvec, tvec = self.pnp.estimate_pose_ransac(camera_matrix, object_points, image_points, max_iterations=ransac_max_iterations)
                    else:
                        success, rvec, tvec = self.pnp.estimate_pose(camera_matrix, object_points, image_points)
                except Exception as e:
                    print("Exception:", e)
                    success = False
            else:
                if use_ransac:
                    success, rvec, tvec = self.pnp.estimate_pose_ransac(camera_matrix, object_points, image_points, max_iterations=ransac_max_iterations)
                else:
                    success, rvec, tvec = self.pnp.estimate_pose(camera_matrix, object_points, image_points)

            if success and use_refine:
                success, rvec, tvec = self.pnp.refine(camera_matrix, object_points, image_points, rvec, tvec)

            if analytics is not None:
                analytics.end_pnp()

            if verbose:
                print(f"Success: {success}, rvec: {rvec}, tvec: {tvec}")

            img = frame

            # plot projected points on to image
            if not success:
                print("Failed to solve PnP")
                analytics.add_failed_frame()
            else:
                if show_frame_pose:
                    axis = np.float32([[0,0,0], [0,0,-1], [0,1,0], [1,0,0]]).reshape(-1,3)
                    axis = 0.15 * axis
                    axis_points, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, None)
                    all_projected_points, _ = cv2.projectPoints(all_object_points, rvec, tvec, camera_matrix, None)

                    axis_points = axis_points.reshape(-1, 2)
                    all_projected_points = all_projected_points.reshape(-1, 2)

                    if analytics is not None:
                        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
                        projected_points = projected_points.reshape(-1, 2)
                        reprojection_error = np.mean(np.linalg.norm(image_points - projected_points, axis=1))
                        analytics.add_reprojection_error(reprojection_error)

                    graphics.plot_points_to_image(img, image_points, all_projected_points, axis_points)
                    graphics.show_image(img)

            if analytics is not None:
                analytics.end_frame()

                if verbose:
                    analytics.print_results()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            analytics.add_output_frame(img_rgb)
            
            if write_output:
                video_out.write(img)

            if cv2.waitKey(1) == 27:
                break

        if analytics is not None:
            analytics.end()

        video.release()
        if write_output:
            video_out.release()
        cv2.destroyAllWindows()