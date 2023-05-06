import numpy as np

def cylindrical_project(img, focal_len):
    height, width, channel = img.shape

    theta = np.arctan2(width / 2, focal_len)
    radius = focal_len
    arc = radius * theta
    result_height, result_width = height, int(2 * arc)
    result = np.zeros((result_height, result_width, channel))
    mask = np.zeros((result_height, result_width))
    def shift(x, y):
        return x - result_width / 2, result_height / 2 - y

    def shift_back(xx, yy):
        return xx + width / 2, height / 2 - yy

    s, f = focal_len, focal_len
    for y in range(result_height):
        for x in range(result_width):
            # x' = s * arctan(x / f) => x = tan(x' / s) * f
            # y' = s * y / (sqrt(xx + ff)) => y = y' * sqrt(x ** 2 + f ** 2) / s
            # origin should be the center of the image
            xx, yy = shift(x, y)
            original_x = f * np.tan(xx / s) 
            original_y = yy * np.sqrt(original_x ** 2 + f ** 2) / s
            j, i = shift_back(original_x, original_y)

            if i < 0 or i > height - 1 or j < 0 or j > width - 1:
                continue

            floor_i, ceil_i = int(np.floor(i)), int(np.ceil(i))
            floor_j, ceil_j = int(np.floor(j)), int(np.ceil(j))
            ul_pixel = img[floor_i, floor_j, :].astype(int) # c11
            ll_pixel = img[ceil_i, floor_j, :].astype(int) # c21
            ur_pixel = img[floor_i, ceil_j, :].astype(int) # c12
            lr_pixel = img[ceil_i, ceil_j, :].astype(int) # c22
            
            """
            interpolation      
            0 - - 0.5 - - 1
                   |
                   |
                   |
                  1.1 
                   |
                   |
            1 - - 1.5 - - 3
            """
            dx, dy = j - int(j), i - int(i)
            upper_diff, lower_diff = ur_pixel - ul_pixel , lr_pixel - ll_pixel
            upper_val, lower_val = ul_pixel + upper_diff * dx, ll_pixel + lower_diff * dx
            final_val = upper_val + (lower_val - upper_val) * dy
            result[y][x] = final_val
            mask[y][x] = 1

    return result.astype("uint8"), mask

if __name__ == "__main__":
    import cv2
    img = cv2.imread('../test_data/parrington/prtn01.jpg')  # queryImage
    pImg, _ = cylindrical_project(img, 706.286)
    cv2.imwrite('fuck88.png', pImg)
