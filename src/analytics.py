import time

class Analytics:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.num_frames = None
        self.failed_frames = None

        self.avg_fps = None
        self.avg_pnp_time = None
        self.avg_feature_match_time = None
        self.avg_reprojection_error = None

        self.fps_history = []
        self.pnp_time_history = []
        self.feature_match_time_history = []
        self.reprojection_error_history = []

        self.output_frames = []
    
    def start(self):
        self.start_time = time.time()
        self.num_frames = 0
        self.failed_frames = 0
        self.avg_fps = 0
        self.avg_pnp_time = 0
        self.avg_feature_match_time = 0
        self.avg_reprojection_error = 0
        self.fps_history = []
        self.pnp_time_history = []
        self.feature_match_time_history = []
        self.reprojection_error_history = []
        self.output_frames = []

    def end(self):
        self.end_time = time.time()

    def start_frame(self):
        self.num_frames += 1
        self.frame_start_time = time.time()

    def end_frame(self):
        self.frame_end_time = time.time()
        fps = 1 / (self.frame_end_time - self.frame_start_time)
        self.avg_fps += (fps - self.avg_fps) / self.num_frames
        self.frame_start_time = None
        self.frame_end_time = None
        self.fps_history.append(fps)
        return fps

    def start_pnp(self):
        self.pnp_start_time = time.time()

    def end_pnp(self):
        self.pnp_end_time = time.time()
        pnp_time = self.pnp_end_time - self.pnp_start_time
        self.avg_pnp_time += (pnp_time - self.avg_pnp_time) / self.num_frames
        self.pnp_start_time = None
        self.pnp_end_time = None
        self.pnp_time_history.append(pnp_time)
        return pnp_time

    def start_feature_match(self):
        self.feature_match_start_time = time.time()

    def end_feature_match(self):
        self.feature_match_end_time = time.time()
        feature_match_time = self.feature_match_end_time - self.feature_match_start_time
        self.avg_feature_match_time += (feature_match_time - self.avg_feature_match_time) / self.num_frames
        self.feature_match_start_time = None
        self.feature_match_end_time = None
        self.feature_match_time_history.append(feature_match_time)
        return feature_match_time

    def add_reprojection_error(self, reprojection_error):
        self.avg_reprojection_error += (reprojection_error - self.avg_reprojection_error) / (self.num_frames - self.failed_frames)
        self.reprojection_error_history.append(reprojection_error)

    def add_failed_frame(self):
        self.failed_frames += 1
        self.reprojection_error_history.append(-1)

    def add_output_frame(self, frame):
        self.output_frames.append(frame)

    def print_results(self):
        print(f"Analytics for {self.name}")
        if self.end_time is None:
            elapsed_time = time.time() - self.start_time
            print(f"Elapsed time: {elapsed_time}s")
        else:
            total_time = self.end_time - self.start_time
            print(f"Total time: {total_time}s")
        print(f"Average FPS: {self.avg_fps}")
        print(f"Average PnP time: {self.avg_pnp_time}s")
        print(f"Average feature match time: {self.avg_feature_match_time}s")
        print(f"Average reprojection error: {self.avg_reprojection_error}")
        print(f"Failed frames: {self.failed_frames}")
