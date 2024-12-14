from src.pose_estimator import PoseEstimator
from src.camera_config.pixel7 import Pixel7
from src.feature_matcher.image_matching_models import ImageMatchingModels
from src.pnp.opencv_pnp import OpenCVPNP
from src.pnp.ippe import IPPE
from src.analytics import Analytics
import cv2

camera_config = Pixel7()

feature_matcher = ImageMatchingModels("sift-lg")
pnp = IPPE()
pose_estimator = PoseEstimator(camera_config, feature_matcher, pnp)
analytics = Analytics("")

pose_estimator.video_estimate_pose("images/base_box.jpg", "videos/video_box_close.mp4", out_folder="sample_outputs", out_name="sift_lightglue_box_close", use_ransac=False, write_output=True, analytics=analytics)
pose_estimator.video_estimate_pose("images/base_box.jpg", "videos/video_box_far.mp4", out_folder="sample_outputs",out_name="sift_lightglue_box_far", use_ransac=False, write_output=True, analytics=analytics)
pose_estimator.video_estimate_pose("images/base_guitar.png", "videos/video_guitar.mp4", out_folder="sample_outputs",out_name="sift_lightglue_guitar", use_ransac=False, write_output=True, analytics=analytics)

pose_estimator.video_estimate_pose("images/base_box.jpg", "videos/video_box_close.mp4", out_folder="sample_outputs",out_name="lucas_kanade_box_close", use_lucas_kanade=True, write_output=True, analytics=analytics)
pose_estimator.video_estimate_pose("images/base_box.jpg", "videos/video_box_far.mp4", out_folder="sample_outputs",out_name="lucas_kanade_box_far", use_lucas_kanade=True, write_output=True, analytics=analytics)
pose_estimator.video_estimate_pose("images/base_guitar.png", "videos/video_guitar.mp4", out_folder="sample_outputs",out_name="lucas_kanade_guitar", use_lucas_kanade=True, write_output=True, analytics=analytics)

feature_matcher = ImageMatchingModels("sift-nn")
pnp = IPPE()
pose_estimator = PoseEstimator(camera_config, feature_matcher, pnp)
analytics = Analytics("")
pose_estimator.video_estimate_pose("images/base_box.jpg", "videos/video_box_close.mp4", out_folder="sample_outputs",out_name="sift_nn_box_close", use_ransac=False, write_output=True, analytics=analytics)

feature_matcher = ImageMatchingModels("orb-nn")
pnp = IPPE()
pose_estimator = PoseEstimator(camera_config, feature_matcher, pnp)
analytics = Analytics("")
pose_estimator.video_estimate_pose("images/base_box.jpg", "videos/video_box_close.mp4", out_folder="sample_outputs",out_name="orb_nn_box_close", use_ransac=False, write_output=True, analytics=analytics)