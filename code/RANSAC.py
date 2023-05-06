import math
import random
import numpy as np

def ransac(kpts_1, kpts_2, matches, thres = 5, round = 3000):
    max_num_inliers = - math.inf
    best_offset_x = None
    best_offset_y = None
    best_model = None
    i = round
    while i:
        i -= 1
        picked_match = random.choice(matches)
        kpt1_idx = picked_match[0]
        kpt2_idx = picked_match[1]
        offset_x = kpts_1[kpt1_idx].x - kpts_2[kpt2_idx].x
        offset_y = kpts_1[kpt1_idx].y - kpts_2[kpt2_idx].y

        num_inliers = 0
        tmp_matches = []
        for match in matches:
            kpt1_idx = match[0]
            kpt2_idx = match[1]
            err = math.pow(kpts_1[kpt1_idx].x - kpts_2[kpt2_idx].x - offset_x, 2) + \
                math.pow(kpts_1[kpt1_idx].y - kpts_2[kpt2_idx].y - offset_y, 2)
            
            if err >= thres:
                continue

            num_inliers += 1
            tmp_matches.append(match)

        if num_inliers <= max_num_inliers:
            continue

        max_num_inliers = num_inliers
        best_offset_x = int(np.round(offset_x))
        best_offset_y = int(np.round(offset_y))
        best_model = tmp_matches
    
    return best_offset_x, best_offset_y, best_model
