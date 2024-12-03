from src.pose_estimator import PoseEstimator
from src.camera_config.pixel7 import Pixel7
from src.feature_matcher.image_matching_models import ImageMatchingModels
from src.pnp.opencv_pnp import OpenCVPNP
from src.pnp.ippe import IPPE
from src.analytics import Analytics
import cv2

camera_config = Pixel7()

# feature_matcher = ImageMatchingModels("sift-nn")
# feature_matcher = ImageMatchingModels("sift-lg")
feature_matcher = ImageMatchingModels("superpoint-lg")
# feature_matcher = ImageMatchingModels("orb-nn")
# feature_matcher = ImageMatchingModels("superglue")
# feature_matcher = ImageMatchingModels("aliked-lg", "cpu")
# feature_matcher = ImageMatchingModels("loftr")

# pnp = OpenCVPNP(cv2.SOLVEPNP_ITERATIVE)
# pnp = OpenCVPNP(cv2.SOLVEPNP_IPPE)
# pnp = OpenCVPNP(cv2.SOLVEPNP_EPNP)
pnp = IPPE()

pose_estimator = PoseEstimator(camera_config, feature_matcher, pnp)

analytics = Analytics()

pose_estimator.video_estimate_pose("images/base_0.jpg", "videos/video_0.mp4", use_ransac=False, verbose=True, analytics=analytics)

print("\nFinal Results:")
analytics.print_results()