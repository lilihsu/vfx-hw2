import os
import sys
import cv2
import time
import json
import queue
import scipy
import pickle
import random
import logging
import threading
import numpy as np
import matplotlib.pyplot as plt
import RANSAC

logging.basicConfig(
    filename="stitch.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG
)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"x: {self.x}, y: {self.y}"
    
def linear_blend(img1, img2, mask1, mask2, dx, dy):
    # Determine the source and destination
    # Always refers to the RHS image as source
    if dx < 0:
        src, dst = img1, img2
        src_mask, dst_mask = mask1, mask2
        dx, dy = -dx, -dy
    else:
        src, dst = img2, img1
        src_mask, dst_mask = mask2, mask1

    # Here dx is always positive but dy is not
    height_s, width_s = src.shape[:2]
    height_d, width_d = src.shape[:2]
    height = height_s + abs(dy)
    width = width_s + dx

    # Shift the image
    # For the destination, this just act as resize
    M_s = np.array([[1, 0, dx], [0, 1, dy]]).astype(np.float32)
    M_d = np.array([[1, 0, 0], [0, 1, 0]]).astype(np.float32)

    t_src, t_src_mask = cv2.warpAffine(src, M_s, (width, height)), cv2.warpAffine(src_mask, M_s, (width, height))
    t_dst, t_dst_mask = cv2.warpAffine(dst, M_d, (width, height)), cv2.warpAffine(dst_mask, M_d, (width, height))

    # Calculate the weight for combining 2 images
    dst_one_width = dx
    overlap_width = dst.shape[1] - dx
    src_zero_width = width - dst.shape[1]
    #print(dx, dy)
    #print(dst_one_width)
    #print(overlap_width)
    #print(src_zero_width)
    dst_weight_row = np.concatenate([
        np.ones(dst_one_width), 
        np.linspace(1, 0, overlap_width), 
        np.zeros(src_zero_width)
    ], axis=0)
    
    dst_weight_2d = np.repeat(np.array([dst_weight_row]), height, axis=0)
    dst_weight_3c = np.repeat(np.expand_dims(dst_weight_2d, axis=2), 3, axis=2)
    src_weight_3c = 1 - dst_weight_3c
    result = dst_weight_3c * t_dst + src_weight_3c * t_src

    dst_not_src = np.logical_and(t_dst_mask, np.logical_not(t_src_mask))
    src_not_dst = np.logical_and(t_src_mask, np.logical_not(t_dst_mask))
    result[dst_not_src] = t_dst[dst_not_src]
    result[src_not_dst] = t_src[src_not_dst]
    result_mask = np.logical_or(t_dst_mask, t_src_mask)

    return result.astype(np.uint8) , result_mask.astype(np.uint8)


def poisson_blend(img1, img2, mask1, mask2, dx, dy):
    # Determine the source and destination
    # Always refers to the RHS image as source
    if dx < 0:
        src, dst = img1, img2
        src_mask, dst_mask = mask2, mask1
        dx, dy = -dx, -dy
    else:
        src, dst = img2, img1
        src_mask, dst_mask = mask1, mask2
    width, height = src.shape[1] + dx, src.shape[0] + abs(dy)

    # Get upper left(ul) and lower right(lr) vertices of the rectangle
    height_d, width_d = dst.shape[:2]
    height_s, width_s = src.shape[:2]
    if dy > 0:
        src_ul, src_lr = Point(0, 0), Point(width_d - dx, height_d - dy)
        dst_ul, dst_lr = Point(dx, dy), Point(width_d, height_d)
        N = (width_d - dx) * (height_d - dy)
    else:
        src_ul, src_lr = Point(0, abs(dy)), Point(width_d - dx, height_s)
        dst_ul, dst_lr = Point(dx, 0), Point(width_d, height_s - abs(dy))
        N = (width_d - dx) * (height_s - abs(dy))

    # First directly merge 2 images, then fills the pixel value within the overlapped area
    src_area = src[src_ul.y:src_lr.y, src_ul.x:src_lr.x, :].copy()
    """
    logging.info(f"dx = {dx}, dy = {dy}")
    logging.info(f"result width = {width}, height = {height}")
    logging.info(f"dst width = {width_d}, height = {height_d}")
    logging.info(f"src width = {width_s}, height = {height_s}")
    logging.info(f"src upper left: {src_ul}")
    logging.info(f"src lower right: {src_lr}")
    logging.info(f"dst upper left: {dst_ul}")
    logging.info(f"dst lower right: {dst_lr}")
    logging.info(f"src area shape: {src_area.shape}")
    """
    M_src = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    M_dst = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    t_src, t_src_mask = cv2.warpAffine(src, M_src, (width, height)), cv2.warpAffine(src_mask, M_src, (width, height))
    t_dst, t_dst_mask = cv2.warpAffine(dst, M_dst, (width, height)), cv2.warpAffine(dst_mask, M_dst, (width, height))
    result = t_src + t_dst
    result_mask = np.logical_or(t_src_mask, t_dst_mask)

    # Build the pixel-to-matrix-index mapping, (x, y) -> vector index
    def build_pixel_index_mapping(ul, lr):
        mapping, reverse_mapping, idx = {}, {}, 0
        for y in range(ul.y, lr.y):
            for x in range(ul.x, lr.x):
                mapping[idx] = Point(x, y)
                reverse_mapping[(x, y)] = idx
                idx += 1
        return mapping, reverse_mapping
    
    idx_to_pixel, pixel_to_idx = build_pixel_index_mapping(dst_ul, dst_lr)

    def in_omega(x, y, ul, lr):
        return ul.x <= x < lr.x and ul.y <= y < lr.y

    def dst_to_src(x, y):
        return x - dx, y - abs(dy)

    def solve_channel(X, channel=0):
        A = scipy.sparse.lil_matrix((N, N), dtype=float)
        b = np.zeros(N, dtype=float)
        
        for i in range(N):
            p = idx_to_pixel[i]
            assert(in_omega(p.x, p.y, dst_ul, dst_lr))
            sx, sy = dst_to_src(p.x, p.y)
            Np = 0
            b_val = 0.0
            direction = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            for d in direction:
                nx, ny = p.x + d[0], p.y + d[1]
                if 0 <= nx < width_d and 0 <= ny < height_d:
                    Np += 1
                    snx, sny = dst_to_src(nx, ny)
                    #gs = src[sy][sx][channel] - src[sny][snx][channel]
                    gd = dst[p.y][p.x][channel] - dst[ny][nx][channel]
                    b_val += gd
                    if in_omega(nx, ny, dst_ul, dst_lr):
                        j = pixel_to_idx[(nx, ny)]
                        A[(i, j)] = -1
                    else:
                        b_val += dst[ny][nx][channel]
            # if Np < 4:
            #    logging.info(f"Np of point p ({p}): {Np}")
            A[(i, i)] = Np
            b[i] = b_val
        
        # x = scipy.sparse.linalg.lsqr(A, b, iter_lim=1500, x0=src_area[:, :, channel].reshape(-1))[0]
        x = scipy.sparse.linalg.cg(A, b, maxiter=10000, x0=src_area[:, :, channel].reshape(-1))[0]
        X[channel] = x
    
    print("Start solving linear system...")
    start = time.time()
    x = [None] * 3
    threads = [threading.Thread(target=solve_channel, args=(x,), kwargs={"channel": i}) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"Solving 3 channels takes {time.time() - start}s")

    for i in range(N):
        pixel = idx_to_pixel[i]
        result[pixel.y][pixel.x][0] = x[0][i]
        result[pixel.y][pixel.x][1] = x[1][i]
        result[pixel.y][pixel.x][2] = x[2][i]

    dst_not_src = np.logical_and(t_dst_mask, np.logical_not(t_src_mask))
    src_not_dst = np.logical_and(t_src_mask, np.logical_not(t_dst_mask))
    result[dst_not_src] = t_dst[dst_not_src]
    result[src_not_dst] = t_src[src_not_dst]
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8), result_mask.astype(np.uint8)

def stitch(imgs, masks, keypoints, matches, matrix, n_candidates=2, start=0, algo="linear"):
    q = queue.Queue()
    q.put(start)

    accu_dx = np.zeros(len(imgs)).astype(int)
    accu_dy = np.zeros(len(imgs)).astype(int)
    used = np.zeros(len(imgs))
    used[start] = 1

    result = imgs[start].copy()
    result_mask = masks[start].copy()

    while not q.empty():
        # i: last stitched image index
        i = q.get()
        img = imgs[i]

        # Pick n_candidates images with largest amount of matches 
        candidates = np.argsort(matrix[i])[::-1][:n_candidates]
        candidate = None 
        for j in candidates:
            if not used[j]:
                candidate = j
                break
        
        if candidate == None:
            print("Something went wrong..")
            break
            
        dx, dy, good = RANSAC.ransac(keypoints[i], keypoints[candidate], matches[i][candidate])
        if len(matches[i][candidate]) > 5.9 + 0.22 * len(good):
            dx += accu_dx[i]
            dy += accu_dy[i]
            if algo == "linear":
                result, result_mask = linear_blend(result, imgs[candidate], result_mask, masks[candidate], dx, dy)
            elif algo == "poisson":
                result, result_mask = poisson_blend(result, imgs[candidate], result_mask, masks[candidate], dx, dy)
            else:
                raise NotImplementedError

            # Update offsets of images have been stitched
            if dx < 0:
                accu_dx[used == 1] += -dx
                accu_dy[used == 1] += -dy
            else:
                accu_dx[candidate] = dx
                accu_dy[candidate] = dy

            # Update status and the queue
            used[candidate] = 1
            q.put(candidate)

    return result
