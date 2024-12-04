from src.pnp.pnp import PNP
import numpy as np
import cv2


class IPPE(PNP):
    def __init__(self):
        super().__init__()

    def normalize_image_points(self,image_points, camera_matrix):
        """
        Convert image points using the camera matrix to normalized coordinates.
        This is the Line2 in algorithm 2
        """
        normalized_points = np.linalg.inv(camera_matrix) @ np.vstack((image_points.T, np.ones(image_points.shape[0])))
        normalized_points = normalized_points[:2] / normalized_points[2]
        return normalized_points.T

    def ippe(self,v, J):
        '''
        Pseudo Code: Algorithm 1 in paper
        '''
        # Line2: Compute the correction rotation Rv
        t = np.linalg.norm(v)
        if t <= np.finfo(float).eps:
            # Plane is fronto-parallel to the camera
            Rv = np.eye(3)
        else:
            # Plane is not fronto-parallel
            s = np.linalg.norm(np.append(v, 1))
            costh = 1 / s
            sinth = np.sqrt(1 - 1 / (s ** 2))

            Kcrs = np.zeros((3, 3))
            Kcrs[:2, 2] = v
            Kcrs[2, :2] = -v
            Kcrs = Kcrs / t
            Rv = np.eye(3) + sinth * Kcrs + (1 - costh) * (Kcrs @ Kcrs)

        #Line3: Setup the 2x2 SVD decomposition
        B = np.hstack((np.eye(2), -v.reshape(2, 1))) @ Rv[:, :2]

        #Line4
        Binv = np.linalg.inv(B)
        A = Binv @ J

        #Line5: Compute the largest singular value of A
        AAT = A @ A.T
        gamma = np.sqrt(0.5 * (AAT[0, 0] + AAT[1, 1] +
                               np.sqrt((AAT[0, 0] - AAT[1, 1]) ** 2 + 4 * AAT[0, 1] ** 2)))

        #Line6: Reconstruct the full rotation matrices
        R22_tild = A / gamma

        #Line7
        h = np.eye(2) - R22_tild.T @ R22_tild
        b = np.array([np.sqrt(h[0, 0]), np.sqrt(h[1, 1])])
        if h[0, 1] < 0:
            b[1] = -b[1]

        #Line8
        d = np.cross(
            np.append(R22_tild[:, 0], b[0]),
            np.append(R22_tild[:, 1], b[1])
        )
        c = d[:2]
        a = d[2]

        #Line9
        R1 = Rv @ np.vstack((np.hstack((R22_tild, c.reshape(2, 1))),
                             np.array([b[0], b[1], a])))
        R2 = Rv @ np.vstack((np.hstack((R22_tild, -c.reshape(2, 1))),
                             np.array([-b[0], -b[1], a])))

        return R1, R2

    def estimate_t(self,R, points_3d, points_2d):
        '''
        The translation vector fitting that is not shown in paper but shown in their github as matlab code
        '''
        points_3d = np.pad(points_3d, ((0, 0), (0, 1)), mode='constant')
        num_points = points_3d.shape[0]
        Ps = R @ points_3d.T

        Ax = np.zeros((num_points, 3))
        bx = np.zeros(num_points)
        Ay = np.zeros((num_points, 3))
        by = np.zeros(num_points)

        Ax[:, 0] = 1
        Ax[:, 2] = -points_2d[:, 0]
        bx = points_2d[:, 0] * Ps[2] - Ps[0]

        Ay[:, 1] = 1
        Ay[:, 2] = -points_2d[:, 1]
        by = points_2d[:, 1] * Ps[2] - Ps[1]

        A = np.vstack((Ax, Ay))
        b = np.hstack((bx, by))

        AtA = A.T @ A
        Atb = A.T @ b

        return np.linalg.solve(AtA, Atb)

    def compute_reprojection_errors(self,R1, R2, t1, t2, points_3d, points_2d):
        def project_points(R, t, points):
            P_cam = R @ points.T
            P_cam = P_cam + t.reshape(3, 1)
            P_cam = P_cam / P_cam[2]
            return P_cam[:2].T

        proj1 = project_points(R1, t1, points_3d)
        proj2 = project_points(R2, t2, points_3d)

        error1 = np.linalg.norm(proj1 - points_2d)
        error2 = np.linalg.norm(proj2 - points_2d)

        return error1, error2

    def perspectiveIPPE(self,object_points, image_points, camera_matrix):
        '''
        Pseudo Code: Algorithm 2 in paper
        '''
        #Line2: Convert image points to normalized coordinates
        normalized_points = self.normalize_image_points(image_points, camera_matrix)

        # Use only x,y coordinates from object points
        planar_points = object_points[:, :2]

        # Preprocessing: Center the model points(shown on their github)
        center = np.mean(planar_points, axis=0)
        centered_points = planar_points - center

        # Line3: Estimate homography
        H,_ = cv2.findHomography(centered_points, normalized_points)

        #Line4-8: Compute Jacobian J
        H = H / H[2, 2]
        J = np.array([[H[0, 0] - H[2, 0] * H[0, 2], H[0, 1] - H[2, 1] * H[0, 2]],
                      [H[1, 0] - H[2, 0] * H[1, 2], H[1, 1] - H[2, 1] * H[1, 2]]])

        #Line9
        v = np.array([H[0, 2], H[1, 2]])

        #Line10
        R1, R2 = self.ippe(v, J)

        #Line11: Compute translation solutions
        t1 = self.estimate_t(R1, centered_points, normalized_points)
        t2 = self.estimate_t(R2, centered_points, normalized_points)

        #Extra: Adjust translations for the centered points(Since I shifted the center at the beginning)
        t1_adj = t1 - R1 @ np.append(center, 0)
        t2_adj = t2 - R2 @ np.append(center, 0)

        # Compute reprojection errors
        err1, err2 = self.compute_reprojection_errors(R1, R2, t1_adj, t2_adj,
                                                 np.column_stack((planar_points, np.zeros(len(planar_points)))),
                                                 normalized_points)

        if err1 <= err2:
            return True,R1, t1_adj
        else:
            return True,R2, t2_adj
    def estimate_pose(self,camera_matrix,object_points,image_points):
        success,r,t = self.perspectiveIPPE(object_points,image_points,camera_matrix)
        rvec,_ = cv2.Rodrigues(r)
        tvec = t.reshape(-1,1)
        return success,rvec,tvec
    def estimate_pose_ransac(self, camera_matrix, object_points, image_points, max_iterations=100,
                             reprojection_error=8.0):
        success, rvec, tvec, _ = cv2.solvePnPRansac(object_points, image_points, camera_matrix, None,
                                                    flags=cv2.SOLVEPNP_IPPE, iterationsCount=max_iterations,
                                                    reprojectionError=reprojection_error)
        return success, rvec, tvec

