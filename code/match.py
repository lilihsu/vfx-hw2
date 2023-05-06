import os
import sys
import cv2
import numpy as np
import matplotlib as plt
import scipy

from detection import harrisDetector
from description import gen_sift_descriptors

np.set_printoptions(threshold=sys.maxsize)

def match_features(keypoints1, keypoints2, descriptors1, descriptors2):
    matches = []
    for i, d1 in enumerate(descriptors1):
        dist = []
        for j, d2 in enumerate(descriptors2):
            dist.append(scipy.spatial.distance.euclidean(d1, d2))
        idx = np.argsort(dist)
        best, second = idx[0], idx[1]
        if dist[best] / dist[second] < 0.8 and abs(keypoints1[i].y - keypoints2[best].y) < 10:
            matches.append((i, best))
    
    return matches

if __name__ == "__main__":
    img1 = cv2.imread("../data/parrington/prtn01.jpg")
    img2 = cv2.imread("../data/parrington/prtn00.jpg")

    key_pnts1 = harrisDetector(img1, k=0.04, threshold_ratio=0.01, window_size=5)
    key_pnts2 = harrisDetector(img2, k=0.04, threshold_ratio=0.01, window_size=5)

    descriptors1 = gen_sift_descriptors(img1, key_pnts1)
    descriptors2 = gen_sift_descriptors(img2, key_pnts2)

    good = match_features(key_pnts1, key_pnts2, descriptors1, descriptors2)
    print(good)
    