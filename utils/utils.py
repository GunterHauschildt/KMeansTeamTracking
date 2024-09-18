import numpy as np
import cv2 as cv


def round_and_clip(x0, y0, x1, y1, image_size):
    x0 = round(x0)
    x1 = round(x1)
    y0 = round(y0)
    y1 = round(y1)
    x0 = max(0, min(x0, image_size[1] - 1))
    x1 = max(0, min(x1, image_size[1] - 1))
    y0 = max(0, min(y0, image_size[0] - 1))
    y1 = max(0, min(y1, image_size[0] - 1))
    return x0, y0, x1, y1


def grow_x0y0x1y1(x0, y0, x1, y1, d, image_size):
    x0 -= d
    y0 -= d
    x1 += d
    y1 += d
    return round_and_clip(x0, y0, x1, y1, image_size)


def rect_to_x0y0x1xy(rect, image_size):
    x0 = rect[0]
    y0 = rect[1]
    x1 = x0 + rect[2]
    y1 = y0 + rect[3]
    return round_and_clip(x0, y0, x1, y1, image_size)


def box_to_x0y0x1y1(box, image_size):
    x0, y0 = box[0]
    x1, y1 = box[1]
    return round_and_clip(x0, y0, x1, y1, image_size)


def crop(image, box, copy=True):
    x0, y0, x1, y1 = box_to_x0y0x1y1(box, image.shape)
    if copy:
        image = image[y0:y1, x0:x1].copy()
    else:
        image = image[y0:y1, x0:x1]
    return image, (x0, y0, x1, y1)


def cv_to_tf(image, size_xy=None, to_gray=False):
    if size_xy is not None:
        image_data = cv.resize(image, size_xy)
    else:
        image_data = image
    if not to_gray:
        image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)
    else:
        image_data = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    image_data = image_data.astype(np.float32)
    image_data /= 255.0
    image_data = np.expand_dims(image_data, axis=0)
    return image_data
