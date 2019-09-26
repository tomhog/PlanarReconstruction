# copyright Hogbox Studios 2019

import os
import os.path
import cv2
import base64
import numpy as np
import torch
import math
from utils.disp import tensor_to_image

instance_ids = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 255],
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

def computeContoursAndBoundries(predict_segmentation, shape):
    instance_image = cv2.resize(np.stack([instance_ids[predict_segmentation]], axis=2), shape, interpolation=cv2.INTER_NEAREST)

    uniqueids, uniqueidcounts = np.unique(instance_image, return_counts=True)

    contour_list = []
    boundry_list = []

    ids_itr = iter(uniqueids)
    next(ids_itr)

    for id in ids_itr:
        maskid = int(id)
        mask = cv2.inRange(instance_image, maskid, maskid)

        #cv2.imwrite('tests/debug/instances' + str(id) + '.jpg', mask)

        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #find the biggest area
        main_contour = max(contours, key = cv2.contourArea)
        epsilon = shape[0] / 25 # *cv2.arcLength(main_contour,True)
        main_contour = cv2.approxPolyDP(main_contour,epsilon,True)

        boundry_points = np.vstack(main_contour).squeeze()

        contour_list.append(main_contour)
        boundry_list.append(boundry_points)

    return contour_list, boundry_list

def computeViewSpaceXYZ(depth, segmentation):
    h, w = 192, 256
    view_positions = np.zeros((h,w, 3))
    for y in range(h):
        for x in range(w):
            ray = K_inv_dot_xy_1[:, y, x]
            X, Y, Z = ray * depth[y, x]
            view_positions[y][x] = [X,Y,Z]
    return view_positions

def computeContourNormals(contours, viewspace_xyz):
    normals = []
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)

        if M["m00"] == 0:
            normals.append(([0.0,0.0,0.0], [0.0,0.0,0.0]))
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        offset = 10

        if (cX+offset) >= viewspace_xyz.shape[1]:
            cX = viewspace_xyz.shape[1] - (offset + 1)

        if (cY+offset) >= viewspace_xyz.shape[0]:
            cY = viewspace_xyz.shape[0] - (offset + 1)

        planep0 = viewspace_xyz[cY][cX]
        planep1 = viewspace_xyz[cY][cX + offset]
        planep2 = viewspace_xyz[cY + offset][cX]
        
        p0to1 = planep1 - planep0
        p0to1 = p0to1 / np.linalg.norm(p0to1)
        p0to2 = planep2 - planep0
        p0to2 = p0to2 / np.linalg.norm(p0to2)
        pnormal = -np.cross(p0to1, p0to2)

        normals.append((pnormal, planep0))
    return normals

def vectorLength(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def normalizedVector(v):
    length = vectorLength(v)
    return [v[0] / length, v[1] / length, v[2] / length]

def vectorDot(v1, v2):
    return (v1[0] * v2[0]) + (v1[1] * v2[1]) + (v1[2] * v2[2])

def angleBetweenVectors(v1, v2):
    v1n = normalizedVector(v1)
    v2n = normalizedVector(v2)
    dot = vectorDot(v1n, v2n)
    return math.acos(dot)

def vectorToAngles(normal):
    radius = 1.0 #math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
    theta = math.atan2(normal[2], normal[0])
    phi = math.acos(normal[1] / radius)
    return [theta, phi]

def pitchVector(v, p):
    cs = math.cos(p)
    sn = math.sin(p)

    return [v[0], v[2] * sn + v[1] * cs, v[2] * cs - v[1] * sn]

def generateVMEnvironmentTemplate(predict_segmentation, segmentation, depth, size, fited_size, input_image_path, input_image_base64, cfg):

    canvas_size = [2346, 1440]

    # get contours and boundires for the two resolutions (must be a better way to do this bit)
    floor_contours, floor_boundries = computeContoursAndBoundries(predict_segmentation, (fited_size[0], fited_size[1]))
    low_contours, low_boundries = computeContoursAndBoundries(predict_segmentation, (size[0], size[1]))

    viewspace_xyz = computeViewSpaceXYZ(depth, segmentation)

    # compute normals for all contour planes
    normals_and_centers = computeContourNormals(low_contours, viewspace_xyz)

    # find the floor contour
    up_vector = np.array([0,1,0])

    lower_quad_contours_indicies = []
    index = 0
    for lb in low_contours:
        for lbp in lb:
            if lbp[0][1] >= (size[1]-(size[1]/4)):
                lower_quad_contours_indicies.append(index)
                break
        index = index + 1

    index_of_floor = 0
    area_of_floor = 0

    for lqci in lower_quad_contours_indicies:
        area = cv2.contourArea(low_contours[lqci])
        if area > area_of_floor: #  abs(nc[0][1]) > y_of_floor:
            index_of_floor = lqci
            area_of_floor = area
    

    floor_boundry_points = floor_boundries[index_of_floor]
    floor_plane_normal = normals_and_centers[index_of_floor][0]
    floor_viewspace_center = normals_and_centers[index_of_floor][1]
    floor_plane_angles_test = vectorToAngles(floor_plane_normal)
    floor_plane_pitch = angleBetweenVectors(floor_plane_normal, [0,0,-1])
    pitched_floor_vector = pitchVector([0,0,-1], floor_plane_pitch)
    floor_plane_roll = angleBetweenVectors(floor_plane_normal, pitched_floor_vector)

    # read the json template and insert out base 64 data
    jsonstream = open("evn_template.json", "r")
    json_template_string = jsonstream.read()
    jsonstream.close()

    base64_header = '$base64^jpeg,'

    json_template_string = json_template_string.replace('MASK_IMAGE_TOKEN', "") # base64_header + mask_image_base64.decode('utf-8'))
    json_template_string = json_template_string.replace('COLOR_IMAGE_TOKEN', base64_header + input_image_base64.decode('utf-8'))

    # fill boundry list
    vector_string_template = "{\"x\": X_VAL,\"y\": Y_VAL}"
    floor_boundry_points_string = ""
    for vec in floor_boundry_points:
        xcoord = (vec[0] - (fited_size[0] * 0.5)) # + (canvas_size[0] * 0.5)
        ycoord = ((canvas_size[1] - vec[1]) - (fited_size[1] * 0.5)) # + (canvas_size[1] * 0.5)
        vec_str = vector_string_template.replace("X_VAL", str(xcoord))
        vec_str = vec_str.replace("Y_VAL", str(ycoord))
        floor_boundry_points_string += vec_str + ","
    floor_boundry_points_string = floor_boundry_points_string[:-1]

    json_template_string = json_template_string.replace("BOUNDRY_POINTS_TOKEN", floor_boundry_points_string)

    
    # compute floor perspective points
    floor_plane_tanget = np.cross(floor_plane_normal, up_vector)
    floor_plane_tanget = floor_plane_tanget / np.linalg.norm(floor_plane_tanget)
    floor_plane_bitanget = np.cross(floor_plane_normal, floor_plane_tanget)
    floor_plane_bitanget = floor_plane_bitanget / np.linalg.norm(floor_plane_bitanget)
    floor_plane_tanget *= 0.5
    floor_plane_bitanget *= 0.5

    floor_pose_points = np.array([
                                    [floor_viewspace_center + (-floor_plane_tanget + -floor_plane_bitanget) ],
                                    [floor_viewspace_center + (floor_plane_tanget + -floor_plane_bitanget) ],
                                    [floor_viewspace_center + (floor_plane_tanget + floor_plane_bitanget) ],
                                    [floor_viewspace_center + (-floor_plane_tanget + floor_plane_bitanget) ]
                                ])
    
    focal_length = float(canvas_size[1] / 2)
    center = (canvas_size[0]/2, canvas_size[1]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    rotation_vector = np.zeros((3,1))
    translation_vector = np.zeros((3,1))

    (projected_floor_pose_points, jacobian) = cv2.projectPoints(floor_pose_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    projected_floor_pose_points = projected_floor_pose_points.reshape(-1, 2)


    perspective_points_string = ""
    for vec in projected_floor_pose_points:
        xcoord = (vec[0] - (canvas_size[0] * 0.5)) # + (canvas_size[0] * 0.5)
        ycoord = ((canvas_size[1] - vec[1]) - (canvas_size[1] * 0.5)) # + (canvas_size[1] * 0.5)
        vec_str = vector_string_template.replace("X_VAL", str(xcoord))
        vec_str = vec_str.replace("Y_VAL", str(ycoord))
        perspective_points_string += vec_str + ","
    perspective_points_string = perspective_points_string[:-1]

    json_template_string = json_template_string.replace("PERSPECTIVE_POINTS_TOKEN", perspective_points_string)

    # fill floor angles
    json_template_string = json_template_string.replace("FLOOR_PLANE_PITCH_TOKEN", str(floor_plane_pitch * 57.2958))
    json_template_string = json_template_string.replace("FLOOR_PLANE_ROLL_TOKEN", str(floor_plane_roll * 57.2958))

    # build output filename
    input_name_w_ext = os.path.basename(input_image_path)
    input_name, input_ext = os.path.splitext(input_name_w_ext)
    input_path, input_name_w_ext = os.path.split(os.path.abspath(input_image_path))

    output_json_file = os.path.join(input_path, input_name + ".json" )

    output_json_stream = open(output_json_file, 'w')
    output_json_stream.write(json_template_string)
    output_json_stream.close()

    #cv2.imshow('image', image)
    #cv2.waitKey(0)

    if cfg.input_image == "test":
        image = cv2.drawContours(tensor_to_image(image.cpu()[0]), low_contours, index_of_floor, 255)
        cv2.imwrite(os.path.join(input_path + '/debug/',  input_name + "-floor.jpg"), image)