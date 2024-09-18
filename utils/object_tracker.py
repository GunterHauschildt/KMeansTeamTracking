import norfair as nf
from norfair.filter import FilterPyKalmanFilterFactory
import cv2 as cv
import numpy as np


class ObjectTracker:
    def __init__(self, args, video_stream):

        distance_function = "euclidean"
        distance_threshold = video_stream.size_xy()[0] * args.tracker_distance_threshold_perc

        self._tracker = nf.Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
            hit_counter_max=args.tracker_timeout_s * video_stream.fps(),
            initialization_delay=args.tracker_timeout_s * video_stream.fps() * .33,
            filter_factory=FilterPyKalmanFilterFactory(R=args.tracker_r, Q=args.tracker_q)
        )

    def tracker(self):
        return self._tracker

    def __repr__(self):
        return ", ".join([self._tracker.__repr__()])

    def __str__(self):
        self.__repr__()


class MotionEstimator(nf.camera_motion.MotionEstimator):
    def __init__(self, size_xy: tuple[int, int], mask_dtype):
        super().__init__()
        self._mask = None
        self._size_xy = size_xy
        self._dtype = mask_dtype
        self.reset()

    def update_mask(self, pt1: tuple[int, int], pt2: tuple[int, int]):
        self._mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = 0

    def reset(self):
        self._mask = np.ones((self._size_xy[1], self._size_xy[0], 1), self._dtype)

    def update(self, frame):
        return super().update(frame, self._mask)



