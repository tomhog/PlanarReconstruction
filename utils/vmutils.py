import os
import os.path
import cv2
import base64
import numpy as np
import torch

instance_ids = np.array([0, 255],
                    dtype=np.uint8)

def get_K_inv_dot_xy_1(h=192, w=256):
    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x],
         [0, focal_length, offset_y],
         [0, 0, 1]]

    K_inv = np.linalg.inv(np.array(K))

    K_inv_dot_xy_1 = np.zeros((3, h, w))

    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640
                
            ray = np.dot(K_inv,
                         np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]

    return K_inv_dot_xy_1


K_inv_dot_xy_1 = get_K_inv_dot_xy_1()


def get_K_dot_xy_1(h=192, w=256):
    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x],
         [0, focal_length, offset_y],
         [0, 0, 1]]

    K = np.array(K)

    K_dot_xy_1 = np.zeros((3, h, w))

    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640
                
            ray = np.dot(K,
                         np.array([xx, yy, 1]).reshape(3, 1))
            K_dot_xy_1[:, y, x] = ray[:, 0]

    return K_dot_xy_1


K_dot_xy_1 = get_K_dot_xy_1()

def computeMainContourAndBoundry(predict_segmentation, shape):
    instance_image = cv2.resize(np.stack([instance_ids[predict_segmentation]], axis=2), shape, interpolation=cv2.INTER_CUBIC)

    im2, contours, hierarchy = cv2.findContours(instance_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #find the biggest area
    main_contour = max(contours, key = cv2.contourArea)
    epsilon = 10 # *cv2.arcLength(main_contour,True)
    main_contour = cv2.approxPolyDP(main_contour,epsilon,True)

    boundry_points = np.vstack(main_contour).squeeze()

    return main_contour, boundry_points

def computeViewSpaceXYZ(depth, segmentation):
    h, w = 192, 256
    view_positions = np.zeros((h,w, 3))
    for y in range(h):
        for x in range(w):
            ray = K_inv_dot_xy_1[:, y, x]
            X, Y, Z = ray * depth[y, x]
            view_positions[y][x] = [X,Y,Z]
    return view_positions

def computeImageSpaceXYZ(viewspace_xyz, segmentation):
    h, w = 192, 256
    view_positions = np.zeros((h,w, 3))
    for y in range(h):
        for x in range(w):
            ray = K_dot_xy_1[:, y, x]
            X, Y, Z = ray * np.linalg.norm(viewspace_xyz)
            view_positions[y][x] = [X,Y,Z]
    return view_positions