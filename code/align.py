import os
import sys
import cv2
import numpy as np

def align_perspective(result):
    height, width = result.shape[:2]
    compress_img = np.sum(result, axis=2)

    left, right = 0, width - 1
    while True:
        left_nonzero = np.where(compress_img[:, left] != 0)
        if len(left_nonzero[0]) > 0:
            break
        left += 1
    while True:
        right_nonzero = np.where(compress_img[:, right] != 0)
        if len(right_nonzero[0]) > 0:
            break
        right -= 1

    l_upper, l_lower = min(left_nonzero[0]), max(left_nonzero[0])
    r_upper, r_lower = min(right_nonzero[0]), max(right_nonzero[0])
    aligned_height = min(l_lower - l_upper, r_lower - r_upper)
    src = np.array([[left, l_upper], [left, l_lower], [right, r_lower], [right, r_upper]], dtype=np.float32)
    dst = np.array([[0, 0], [0, aligned_height - 1], [right, aligned_height - 1], [right, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    aligned_result = cv2.warpPerspective(result, M, (width, aligned_height))
    return aligned_result

if __name__ == "__main__":
    img = cv2.imread("./fuck3.jpg")
    img = align_perspective(img)
    cv2.imwrite("fuck4.jpg", img)
