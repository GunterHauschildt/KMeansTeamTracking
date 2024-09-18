import cv2 as cv
import numpy as np
from numpy.typing import NDArray


class KeyPoints:
    points_by_name = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    lines_by_name = [
        ["left_ear", "left_eye", "nose", "right_eye", "right_ear"],
        ["left_wrist", "left_elbow", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist"],
        ["left_ankle", "left_knee", "left_hip", "right_hip", "right_knee", "right_ankle"]
    ]

    lines = None
    point_confidence = .25

    @staticmethod
    def init_keypoint_lines():
        if KeyPoints.lines is None:
            KeyPoints.lines = []
            for line in KeyPoints.lines_by_name:
                KeyPoints.lines.append(
                    [KeyPoints.points_by_name.index(point) for point in line]
                )

    def __init__(self,
                 frame: NDArray,
                 name: int or str,
                 box: NDArray,
                 xys: NDArray,
                 confs: NDArray
                 ):

        KeyPoints.init_keypoint_lines()
        self._frame = frame
        self._name = str(name)
        self._box = box
        self._xys = xys
        self._confs = confs.copy()

        for i, xy in enumerate(self._xys):
            if self._confs[i] < KeyPoints.point_confidence or \
               self._xys[i][0] <= 5 or self._xys[i][1] <= 5:
                self._confs[i] = -1.0

    def draw(self, m: NDArray, color: tuple[int, int, int]) -> NDArray:

        scale_xy = np.array([m.shape[1] / self._frame.shape[1], m.shape[0] / self._frame.shape[0]])

        for keypoint_line in KeyPoints.lines:

            # draw a line and draw circles.
            # draw circles in its own loop in particular in case the end points are good
            # but the next-to-end points are bad

            def scale_points(xy: NDArray):
                return np.round(scale_xy * xy).astype(np.int32)

            for i in range(len(keypoint_line) - 1):
                if self._confs[keypoint_line[i]] > 0 and self._confs[keypoint_line[i + 1]] > 0:
                    x0y0 = scale_points(self._xys[keypoint_line[i]])
                    x1y1 = scale_points(self._xys[keypoint_line[i + 1]])
                    m = cv.line(m, x0y0, x1y1, color, 3)

            for i in range(len(keypoint_line)):
                if self._confs[keypoint_line[i]] > 0:
                    x0y0 = scale_points(self._xys[keypoint_line[i]])
                    m = cv.circle(m, x0y0, 3, color, 3)

        return m

    def body(self):

        body_pnts = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

        min_conf = min(float(self._confs[KeyPoints.points_by_name.index(pnt)]) for pnt in body_pnts)
        if min_conf < 0:
            return None, None

        tl_x = min(float(self._xys[KeyPoints.points_by_name.index(pnt)][0]) for pnt in body_pnts)
        tl_y = min(float(self._xys[KeyPoints.points_by_name.index(pnt)][1]) for pnt in body_pnts)
        br_x = max(float(self._xys[KeyPoints.points_by_name.index(pnt)][0]) for pnt in body_pnts)
        br_y = max(float(self._xys[KeyPoints.points_by_name.index(pnt)][1]) for pnt in body_pnts)
        body_box = np.array([[tl_x, tl_y], [br_x, br_y]]).astype(np.int32)
        roi = self._frame[body_box[0][1]:body_box[1][1], body_box[0][0]:body_box[1][0]]
        return body_box, roi

