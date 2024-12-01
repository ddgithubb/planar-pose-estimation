import cv2;
import os;
import numpy as np;
import matplotlib.pyplot as plt;
import scipy.signal as signal;

root = os.path.dirname(os.path.abspath(__file__))

target = 0

images_folder = os.path.join(root, "images")
videos_folder = os.path.join(root, "videos")
out_folder = os.path.join(root, "out")

base_img_path = os.path.join(images_folder, f"base_{target}.jpg")
video_path = os.path.join(videos_folder, f"video_{target}.mp4")

# pixel 7
focal_length_mm = 25
sensor_diagonal_inch = 1/1.31
sensor_aspect_ratio = 4/3

lowes_ratio = 0.7
knn_param = 2

def calc_sensor_width(sensor_diagonal_inch, sensor_aspect_ratio):
    sensor_aspect_ratio = 1/sensor_aspect_ratio
    sensor_diagonal_mm = sensor_diagonal_inch * 25.4
    sensor_width_mm = sensor_diagonal_mm / (sensor_aspect_ratio**2 + 1)**0.5
    return sensor_width_mm

def calc_camera_matrix(focal_length_mm, sensor_width_mm, height, width):
    focal_length_pixel = focal_length_mm * width / sensor_width_mm
    camera_matrix = np.array([[focal_length_pixel, 0, width/2], [0, focal_length_pixel, height/2], [0, 0, 1]])
    return camera_matrix

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

def image_pose_detection():
    feature_detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    
    # feature_detector = cv2.ORB_create()
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    sensor_width_mm = calc_sensor_width(sensor_diagonal_inch, sensor_aspect_ratio)

    base_img = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE)
    (height, width) = base_img.shape
    camera_matrix = calc_camera_matrix(focal_length_mm, sensor_width_mm, height, width)
    base_points, base_point_descriptors = plane_points_from_image(feature_detector, base_img, camera_matrix)

    for image in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        (height, width) = img.shape
        camera_matrix = calc_camera_matrix(focal_length_mm, sensor_width_mm, height, width)

        keypoints, descriptors = feature_detector.detectAndCompute(img, None)
  
        matches = matcher.knnMatch(base_point_descriptors, descriptors, 2)
        object_points = []
        image_points = []
        for m, n in matches:
            if m.distance < lowes_ratio * n.distance:
                object_points.append(base_points[m.queryIdx])
                image_points.append(keypoints[m.trainIdx].pt)

        object_points = np.array(object_points)
        image_points = np.array(image_points)

        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None)
        # success, rvec, tvec, _ = cv2.solvePnPRansac(object_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
        # rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, None, rvec, tvec)

        axis = np.float32([[0,0,0], [0,0,-1], [0,1,0], [1,0,0]]).reshape(-1,3)
        axis = 0.1 * axis
        axis_points, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, None)

        # plot projected points on to image
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
        if not success:
            print("Failed to solve PnP")
            projected_points = None
            axis_points = None
        img = cv2.imread(image_path)
        plot_points_to_image(img, image_points, projected_points, axis_points)
        show_image(img)
        cv2.waitKey(0)

def video_pose_detection(write_output=False, pnp_flag=cv2.SOLVEPNP_ITERATIVE, use_extrinsic_guess=False):
    feature_detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    sensor_width_mm = calc_sensor_width(sensor_diagonal_inch, sensor_aspect_ratio)

    base_img = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE)
    (height, width) = base_img.shape
    camera_matrix = calc_camera_matrix(focal_length_mm, sensor_width_mm, height, width)
    base_points, base_point_descriptors = plane_points_from_image(feature_detector, base_img, camera_matrix)
    

    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    if write_output:
        video_out = cv2.VideoWriter(f'{out_folder}/output_{target}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (frame.shape[1], frame.shape[0]))
    
    prev_rvec = None
    prev_tvec = None
    while True:
        ret, frame = video.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = img.shape
        camera_matrix = calc_camera_matrix(focal_length_mm, sensor_width_mm, height, width)

        keypoints, descriptors = feature_detector.detectAndCompute(img, None)
  
        matches = matcher.knnMatch(base_point_descriptors, descriptors, 2)
        object_points = []
        image_points = []
        for m, n in matches:
            if m.distance < lowes_ratio * n.distance:
                object_points.append(base_points[m.queryIdx])
                image_points.append(keypoints[m.trainIdx].pt)

        object_points = np.array(object_points)
        image_points = np.array(image_points)

        print(f"Object points: {object_points.shape}, Image points: {image_points.shape}")

        useExtrinsicGuess = prev_rvec is not None and prev_tvec is not None and use_extrinsic_guess
        
        try:
            # success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None, prev_rvec, prev_tvec, useExtrinsicGuess, flags=pnp_flag)
            success, rvec, tvec, _ = cv2.solvePnPRansac(object_points, image_points, camera_matrix, None, prev_rvec, prev_tvec, useExtrinsicGuess, flags=pnp_flag)
            # rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, None, rvec, tvec)
        except Exception as e:
            print("Exception occured, continuing...")
            success = False

        prev_rvec = rvec
        prev_tvec = tvec

        axis = np.float32([[0,0,0], [0,0,-1], [0,1,0], [1,0,0]]).reshape(-1,3)
        axis = 0.1 * axis
        axis_points, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, None)

        # plot projected points on to image
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
        if not success:
            print("Failed to solve PnP")
            projected_points = None
            axis_points = None

        img = frame
        plot_points_to_image(img, image_points, projected_points, axis_points)
        show_image(img)

        if write_output:
            video_out.write(img)

        if cv2.waitKey(1) == 27:
            break

    video.release()
    if write_output:
        video_out.release()
    cv2.destroyAllWindows()

def main():    
    # video_pose_detection(True, cv2.SOLVEPNP_ITERATIVE, False)
    video_pose_detection(True, cv2.SOLVEPNP_IPPE, False)
    # video_pose_detection(True, cv2.SOLVEPNP_SQPNP, False)

if __name__ == "__main__":
    main()