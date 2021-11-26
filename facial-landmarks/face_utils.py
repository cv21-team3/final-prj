import numpy as np

N_LANDMARKS = 68

# Convert face detected bounding box from dlib
# to OpenCV format (x, y, w, h)
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


# Convert dlib face landmark detector returned shape obj
# to numpy
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((N_LANDMARKS, 2), dtype=dtype)

    for i in range(N_LANDMARKS):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords
