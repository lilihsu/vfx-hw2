import os
import cv2
import time
import argparse
import itertools
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt

from warp import cylindrical_project
from detection import harrisDetector
from description import gen_sift_descriptors
from match import match_features
from stitch import stitch
from align import align_perspective

def wrapper1(img):
    keypoint = harrisDetector(img, k=0.04, threshold_ratio=0.01, window_size=5)
    descriptor = gen_sift_descriptors(img, keypoint)
    return keypoint, descriptor

def wrapper2(keypoints1, keypoints2, descriptors1, descriptors2, i, j):
    return [] if i == j else match_features(keypoints1, keypoints2, descriptors1, descriptors2)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="Input directory", default="../data/denny", dest="input")
    parser.add_argument("-o", "--output-dir", help="Output directory", default="../output", dest="output")
    parser.add_argument("-f", "--focal", help="Camera focal length", default=705, type=float)
    parser.add_argument("-s", "--scale", help="Scale of image resizing", default=0.5, type=float)
    parser.add_argument("-b", "--blend", help="Specifies the blending method", default="linear", type=str)
    return parser

def main():
    parser = arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Input directory does not exist, please check your path.")
        exit(1)
    if not os.path.exists(args.output):
        print("Output directory does not exist, creating one...")
        os.mkdir(args.output)

    # read images
    print("Reading images...")
    imgs = []
    ok_ext = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
    files = sorted(os.listdir(args.input))
    for f in files:
        ext = f[f.rfind(".")+1:]
        if ext not in ok_ext:
            continue
        img = cv2.imread(os.path.join(args.input, f))
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
        imgs.append(img)

    print("Performing cylindrical projection...")
    t_start =  time.time()
    masks = []
    for i, img in enumerate(imgs):
        imgs[i], mask = cylindrical_project(imgs[i], args.focal)
        masks.append(mask)
    print(f"Cylindrical projection takes {time.time() - t_start}s.")

    # Compute SIFT keypoints and descriptors
    print("Calculating keypoints and descriptors...")
    N = len(imgs)
    pool = multiprocessing.Pool(N)
    keypoints = []
    descriptors = []
    t_start = time.time()
    results = pool.map(wrapper1, [imgs[i] for i in range(N)])
    pool.close()
    pool.join()
    for result in results:
        keypoints.append(result[0])
        descriptors.append(result[1])
    assert(len(keypoints) == N and len(descriptors) == N)
    print(f"Calculation of keypoints and descriptors takes {time.time() - t_start}s.")

    print("Matching features...")
    t_start = time.time()
    matching = []
    matrix = np.zeros((len(imgs), len(imgs)), dtype=int)

    for i in range(len(imgs)):
        match_row = []
        pool = multiprocessing.Pool(len(imgs))
        results = pool.starmap(wrapper2, [(keypoints[i], keypoints[j], descriptors[i], descriptors[j], i, j) for j in range(len(imgs))])
        for j, result in enumerate(results):
            match_row.append(result)
            matrix[i, j] = len(result)
        matching.append(match_row)
    print(f"Feature matching takes {time.time() - t_start}s.")

    print("Performing stitching...")
    t_start = time.time()
    paranoma = stitch(imgs, masks, keypoints, matching, matrix, algo=args.blend)
    print(f"Image stitching takes {time.time() - t_start}s.")
    cv2.imwrite(os.path.join(args.output, f"result_{args.blend}.png"), paranoma)
    aligned = align_perspective(paranoma)
    cv2.imwrite(os.path.join(args.output, f"aligned_result_{args.blend}.png"), aligned)

if __name__ == '__main__':
    main()
