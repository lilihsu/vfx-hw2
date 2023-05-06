import cv2
import numpy as np
import math
import detection
import sys
np.set_printoptions(threshold=sys.maxsize)

class Window:
    def __init__(self, row, col, magnitude, orientation):
        self.num_bins = 8
        self.row = row
        self.col = col
        self.magnitude = magnitude
        self.orientation = orientation
    
    def integral_part(self):
        arr = np.array([self.row, self.col, self.magnitude, self.orientation])
        y1, y2 = np.modf(arr)
        y2 = y2.astype(int)
        for i, y in enumerate(y2):
            if arr[i] < 0:
                y2[i] -= 1
        return y2

    def fractional_part(self):
        arr = np.array([self.row, self.col, self.magnitude, self.orientation])
        y1, y2 = np.modf(arr)
        for i, y in enumerate(y2):
            if arr[i] < 0:
                y1[i] += 1
        return y1

def degree_to_rad(degree):
    return (degree / 180.0) * np.pi

def rad_to_degree(rad):
    return (rad / np.pi) * 180.0

def gen_sift_descriptor(img_layer, kpt, scale):
    # Hyperparameters from opencv implementation
    s = 3
    window_size = 4
    num_bins = 8
    max_val = 0.2
    half_size = 8

    row, col = img_layer.shape
    p = np.round(scale * np.array(kpt.pt)).astype(int)
    comp_angle = 360. - kpt.angle
    cos, sin = np.cos(degree_to_rad(comp_angle)), np.sin(degree_to_rad(comp_angle))
    histogram_size = 0.5 * s * scale * kpt.size
    half_size = int(min(np.sqrt(row ** 2 + col ** 2), 0.5 * np.round(histogram_size * (2 ** 0.5) * (window_size + 1))))
    w = -2.0 / (window_size ** 2)
    windows = []

    histogram = np.zeros((window_size + 2, window_size + 2, num_bins))
    for r in range(-half_size, half_size + 1):
        for c in range(-half_size, half_size + 1):
            rotation_matrix = np.array([[cos, sin], [-sin, cos]])
            rot = rotation_matrix @ np.array([r, c])
            idx = rot / histogram_size + 0.5 * window_size - 0.5
            row_idx, col_idx = idx[0], idx[1]
            window_row, window_col = int(np.round(p[1] + r)), int(np.round(p[0] + c))
            if -1 < row_idx < window_size and -1 < col_idx < window_size and \
               0 < window_row < row - 1 and 0 < window_col < col - 1:
                dx = img_layer[window_row][window_col + 1] - img_layer[window_row][window_col - 1]
                dy = img_layer[window_row - 1][window_col] - img_layer[window_row + 1][window_col]
                grad_magnitude = (dx ** 2 + dy ** 2) ** 0.5
                grad_orientation = rad_to_degree(np.arctan2(dy, dx)) % 360
                W = np.exp(w * ((rot[1] / histogram_size) ** 2 + (rot[0] / histogram_size) ** 2))
                windows.append(Window(row_idx, col_idx, W * grad_magnitude, (grad_orientation - comp_angle) * (num_bins / 360)))
        
    def in_bound(pos):
        r, c, o = pos
        return 0 <= r < window_size and 0 <= c < window_size and 0 <= o < window_size

    for window in windows:
        integer = window.integral_part()
        fraction = window.fractional_part()

        r, c, o = np.mgrid[0:2, 0:2, 0:2]
        value = np.array([
            [1 - fraction[0], 1 - fraction[1], 1 - fraction[3]],
            [fraction[0], fraction[1], fraction[3]]
        ]) # row, col, orientation
        for i, j, k in zip(r.flatten(), c.flatten(), o.flatten()):
            val = window.magnitude * value[i][0] * value[j][1] * value[k][2]
            target = (int(integer[0] + 1 + i), int(integer[1] + 1 + j), int((integer[3] + k)) % num_bins)
            histogram[target] += val

    descriptor = histogram[1:-1, 1:-1, :].flatten()
    t = np.linalg.norm(descriptor) * max_val
    descriptor[descriptor > t] = t
    descriptor /= np.linalg.norm(descriptor)
    descriptor = np.clip(np.round(descriptor * 512), 0, 255)
    return descriptor

def gen_sift_descriptors(img, kpts, sigma=1.6, baseSigma=0.5, num_layers=3):
    # generate the base image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.astype(np.float32)
    gray_img = cv2.resize(gray_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_LINEAR)
    diff_sigma = (max(sigma ** 2 - (baseSigma ** 2) * 4, 0.01)) ** (1/2)
    base_img = cv2.GaussianBlur(gray_img, (0, 0), sigmaX=diff_sigma, sigmaY=diff_sigma)
    # print(diff_sigma)
    # print(base_img)
    # print(base_img.shape)

    # calculate octaves
    minEdge = min(base_img.shape)
    num_octaves = int(np.round(math.log2(minEdge) - 1))

    k = math.pow(2, (1. / num_layers))
    gussian_sigmas = []
    gussian_sigmas.append(sigma)
    num_imgs_per_layer = 3 + num_layers

    # generate the sigma of each image
    for img_idx in range(1, num_imgs_per_layer):
        prev_sigma = math.pow(k, img_idx - 1) * gussian_sigmas[0]
        total_sigma = k * prev_sigma
        gussian_sigmas.append((total_sigma ** 2 - prev_sigma ** 2) ** 0.5)

    g_imgs_layers = []
    tmp_g_img = base_img.copy()
    for layer_idx in range(num_octaves):
        g_imgs_per_layer = []
        g_imgs_per_layer.append(tmp_g_img)

        for g_sigma in gussian_sigmas[1:]:
            tmp_g_img = cv2.GaussianBlur(tmp_g_img, (0, 0), sigmaX=g_sigma, sigmaY=g_sigma)
            g_imgs_per_layer.append(tmp_g_img)
        tmp_g_img = cv2.resize(g_imgs_per_layer[-3], None, fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
        g_imgs_layers.append(g_imgs_per_layer)
    g_imgs_layers = np.array(g_imgs_layers, dtype = object)
    # generate the cv2 format keypoint
    cv2_kpts = []
    for kpt in kpts:
        cv2_kpt = cv2.KeyPoint(float(kpt.x), float(kpt.y), size=10)
        cv2_kpts.append(cv2_kpt)

    kpt_descriptors = []

    for cv2_kpt in cv2_kpts:
        layer = 255 & (cv2_kpt.octave >> 8)
        octave = 255 & cv2_kpt.octave 
        if octave >= 128:
            octave = -128 | octave
        
        scale = np.float32(1 << -octave) 
        if octave >= 0: 
            scale = 1 / np.float32(1 << octave)
        # print(layer, octave, scale, sep=',')

        g_img_layer = g_imgs_layers[octave + 1, layer]
        kpt_descriptors.append(gen_sift_descriptor(g_img_layer, cv2_kpt, scale))
    # print(numOfOctaves)
    return kpt_descriptors

if __name__ == '__main__':
    img1 = cv2.imread("../data/parrington/prtn00.jpg")
    key_points = detection.harrisDetector(img1, window_size=5, k=0.04, threshold_ratio=0.01)
    fuck = gen_sift_descriptors(img1, key_points)
    print(fuck)

    