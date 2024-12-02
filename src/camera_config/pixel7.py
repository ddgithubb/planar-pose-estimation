from src.camera_config.camera_config import CameraConfig

class Pixel7(CameraConfig):
    def __init__(self):
        focal_length_mm = 25
        sensor_diagonal_inch = 1/1.31
        sensor_aspect_ratio = 4/3
        super().__init__(focal_length_mm, sensor_diagonal_inch, sensor_aspect_ratio)
        