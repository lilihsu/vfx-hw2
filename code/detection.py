import os
import sys
import cv2
import numpy as np
from scipy.ndimage import maximum_filter

class KeyPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def harrisDetector(img, k=0.04, threshold_ratio=0.01, window_size=5):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32)

    Ix, Iy = np.gradient(cv2.GaussianBlur(img_gray, (window_size, window_size), 0))

    Ix_2 = Ix ** 2
    Iy_2 = Iy ** 2
    Ixy = Ix * Iy

    Sx_2 = cv2.GaussianBlur(Ix_2, (window_size, window_size), 0)
    Sy_2 = cv2.GaussianBlur(Iy_2, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)
    
    responses = (Sx_2 * Sy_2 - Sxy * Sxy) - k * ((Sx_2 + Sy_2) ** 2)

    max_response = np.max(responses)
    thredshold = max_response * threshold_ratio
    for r in np.nditer(responses, op_flags = ['readwrite']):
        if (r < thredshold):
            r[...] = 0.0
    
    local_maximas = maximum_filter(responses, size = 3, mode = 'constant')
    responses[responses < local_maximas] = 0.0
    
    pnts = np.where(responses > 0)
    pnts = np.array(pnts).T
    results = []

    for p in pnts:
        results.append(KeyPoint(int(p[1]), int(p[0])))

    return results

if __name__ == '__main__':
    img1 = cv2.imread("../data/grail/grail00.jpg")
    keypoints = harrisDetector(img1)
    print(keypoints)
