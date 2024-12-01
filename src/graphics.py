import cv2
import numpy as np

def plane_points_from_image(feature_detector, img, camera_matrix):
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    
    # turn keypoints into a plane in 3D space using camera matrix
    points = np.zeros((len(keypoints), 3))
    for i, kp in enumerate(keypoints):
        points[i] = np.array([kp.pt[0], kp.pt[1], 0])
    
    # use camera matrix to convert 2D points to 3D points

    points = np.dot(np.linalg.inv(camera_matrix), points.T).T

    # center the points
    points = points - np.mean(points, axis=0)

    return points, descriptors

def plot_points_to_image(img, keypoints, projected_points, axis_points):
    if projected_points is not None:
        for point in projected_points:
            if np.isnan(point).any() == False:
                cv2.circle(img, (int(point[0][0]), int(point[0][1])), 20, (0, 0, 255), -1)
    for point in keypoints:
        cv2.circle(img, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)
    if axis_points is not None:
        if np.isnan(axis_points).any() == False:
            # draw axis based on the axis points using rgb color
            cv2.line(img, (int(axis_points[0][0][0]), int(axis_points[0][0][1])), (int(axis_points[1][0][0]), int(axis_points[1][0][1])), (0, 0, 255), 30)
            cv2.line(img, (int(axis_points[0][0][0]), int(axis_points[0][0][1])), (int(axis_points[2][0][0]), int(axis_points[2][0][1])), (0, 255, 0), 30)
            cv2.line(img, (int(axis_points[0][0][0]), int(axis_points[0][0][1])), (int(axis_points[3][0][0]), int(axis_points[3][0][1])), (255, 0, 0), 30)

def show_image(img):
    aspect_ratio = img.shape[1] / img.shape[0]
    width = 800
    height = int(width / aspect_ratio)

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", width, height)
    
    cv2.imshow("Result", img)
