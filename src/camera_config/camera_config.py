import numpy as np

class CameraConfig:
    def __init__(self, focal_length_mm, sensor_diagonal_inch, sensor_aspect_ratio):
        self.focal_length_mm = focal_length_mm
        self.sensor_diagonal_inch = sensor_diagonal_inch
        self.sensor_aspect_ratio = sensor_aspect_ratio

    def calc_sensor_width(self):
        sensor_aspect_ratio = 1 / self.sensor_aspect_ratio
        sensor_diagonal_mm = self.sensor_diagonal_inch * 25.4
        sensor_width_mm = sensor_diagonal_mm / (sensor_aspect_ratio**2 + 1)**0.5
        return sensor_width_mm

    def calc_camera_matrix(self, height, width):
        sensor_width_mm = self.calc_sensor_width()
        focal_length_pixel = self.focal_length_mm * width / sensor_width_mm
        camera_matrix = np.array([[focal_length_pixel, 0, width / 2], [0, focal_length_pixel, height / 2], [0, 0, 1]])
        return camera_matrix