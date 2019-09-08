import cv2
import numpy as np
from PIL import Image
import os
from configparser import ConfigParser

config = ConfigParser()
config.read('CONFIG.ini')

input_path = config.get('path_config', 'input_path')
output_path = config.get('path_config', 'output_path')
lb = config.getint('blue_range', 'lb')
lg = config.getint('blue_range', 'lg')
lr = config.getint('blue_range', 'lr')
hb = config.getint('blue_range', 'hb')
hg = config.getint('blue_range', 'hg')
hr = config.getint('blue_range', 'hr')
contour_h_thresh = config.getint('misc', 'contour_h_thresh')
allowed_extension = config.get('misc', 'allowed_extension').strip('')


def hw_text_extraction(path):
    print(path)
    formArr = cv2.imread(path)
    formArr_hsv = cv2.cvtColor(formArr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(formArr_hsv, (lb, lg, lr), (hb, hg, hr))
    hw = cv2.bitwise_and(formArr_hsv,formArr_hsv, mask=blue_mask)
    hw_gray = cv2.cvtColor(hw, cv2.COLOR_BGR2GRAY)
    hw_th = cv2.threshold(hw_gray, 0, 255, cv2.THRESH_OTSU)[1]
    edge_image = cv2.Canny(hw_th, 50, 150)
    kernel = np.ones((2, 60), np.uint8)
    img_dilation = cv2.dilate(edge_image, kernel, iterations=1)
    th, img_th = cv2.threshold(img_dilation, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    f_fill_img = img_th.copy()
    h, w = img_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(f_fill_img, mask, (0, 0), 255)
    f_fill_img_inv = cv2.bitwise_not(f_fill_img)
    connectivity = 8
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(f_fill_img_inv, connectivity)
    label_range = range(1, retval)
    stats = sorted(stats, key=lambda a: a[0])
    bb_img = formArr.copy()

    if not os.path.exists(output_path + os.path.basename(path).split('.')[0] + '/contours'):
        os.makedirs(output_path + os.path.basename(path).split('.')[0] + '/contours', exist_ok=True)
    if not os.path.exists(output_path + os.path.basename(path).split('.')[0] + '/BB'):
        os.makedirs(output_path + os.path.basename(path).split('.')[0] + '/BB', exist_ok=True)

    count = 1
    for label in label_range:
        x, y, w, h, size = stats[label]

        if h < contour_h_thresh:
            continue

        crop_bb = formArr[y:y+h, x:x+w]
        bb_img = cv2.rectangle(bb_img, (x, y), (x + w, y + h), (0, 0, 0), 2)

        cv2.imwrite(output_path + os.path.basename(path).split('.')[0] + '/contours/extraction_' + str(count) + '.jpg', crop_bb)
        count += 1    

    cv2.imwrite(output_path + os.path.basename(path).split('.')[0] + '/BB/contours.jpg', bb_img)

{hw_text_extraction(input_path + input_file) for input_file in os.listdir(input_path) if input_file.endswith(tuple(allowed_extension))}