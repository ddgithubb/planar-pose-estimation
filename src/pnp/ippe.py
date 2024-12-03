from src.pnp.pnp import PNP
import numpy as np
import cv2

class IPPE(PNP):
    def __init__(self):
        super().__init__()

    def estimate_pose(self, camera_matrix, object_points, image_points):
        # Step 1: Compute homography H from object points to image points
        H, _ = cv2.findHomography(object_points, image_points, method=0)

        # Step 2: Normalize homography H such that H33 = 1
        H /= H[2, 2]

        # Step 3: Compute rotation and translation directly using IPPE approach
        # Extracting columns of homography matrix
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        # Calculate the scale factor gamma
        gamma = 1 / np.linalg.norm(np.dot(np.linalg.inv(camera_matrix), h1))

        # Calculate the rotation matrix R
        r1 = gamma * np.dot(np.linalg.inv(camera_matrix), h1)
        r2 = gamma * np.dot(np.linalg.inv(camera_matrix), h2)
        r3 = np.cross(r1, r2)
        R = np.vstack((r1, r2, r3)).T

        # Ensure R is a valid rotation matrix using SVD
        U, _, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)
        rvec, _ = cv2.Rodrigues(R)
        # Calculate the translation vector t
        t = gamma * np.dot(np.linalg.inv(camera_matrix), h3)

        return True, rvec, t

# Example Usage
# Assuming object_points and image_points are np.arrays with appropriate dimensions
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# ippe = IPPE()
# result, rvec, tvec = ippe.estimate_pose(camera_matrix, object_points, image_points)