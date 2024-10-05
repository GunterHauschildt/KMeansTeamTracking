import cv2 as cv


class VideoStream:
    def __init__(self, path: str):
        self._name = path
        self._video_capture = cv.VideoCapture(path)
        self._x = int(self._video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self._y = int(self._video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._ch = 3  # TODO add support for multichannel
        self._fps = self._video_capture.get(cv.CAP_PROP_FPS)
        self._frames = int(self._video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        self._frame_num = 0

    def is_open(self):
        return self._video_capture.isOpened()

    def size_xy(self):
        return self._x, self._y

    def ch(self):
        return self._ch

    def fps(self):
        return self._fps

    def mspf(self):
        return round(1000. / self._fps)

    def next_frame(self):
        success, frame = self._video_capture.read()
        if not success:
            return None
        else:
            frame_num = self._frame_num
            self._frame_num += 1
            return frame, frame_num

    def total_frames(self):
        return self._frames

    def reset(self):
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        success, frame = self._video_capture.read()
        if not success:
            return None
        else:
            self._video_capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            return 0, frame

    def name(self):
        return self._name

    def __repr__(self):
        return ", ".join([self._name, self._x, self._y])

    def __str__(self):
        self.__repr__()
