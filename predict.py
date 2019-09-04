import os
import os.path
import cv2
import base64
import time
import random
import pickle
import numpy as np
from PIL import Image
from distutils.version import LooseVersion

from sacred import Experiment
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torchvision.transforms as tf

from models.baseline_same import Baseline as UNet
from utils.disp import tensor_to_image
from utils.disp import colors as colors
from utils.disp import aspectFitSizeInSize as aspectFitSizeInSize
from bin_mean_shift import Bin_Mean_Shift
from modules import k_inv_dot_xy1
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss

from utils.vmutils import computeMainContourAndBoundry as computeMainContourAndBoundry
from utils.vmutils import computeViewSpaceXYZ as computeViewSpaceXYZ
from utils.vmutils import computeImageSpaceXYZ as computeImageSpaceXYZ

ex = Experiment()

transforms = tf.Compose([
    tf.ToTensor(),
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@ex.command
def eval(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join('experiments', str(_run._id), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # build network
    network = UNet(cfg.model)

    if not cfg.resume_dir == 'None':
        model_dict = torch.load(cfg.resume_dir, None if torch.cuda.is_available() else 'cpu')
        network.load_state_dict(model_dict)

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)
    network.eval()

    bin_mean_shift = Bin_Mean_Shift()
    instance_parameter_loss = InstanceParameterLoss()

    h, w = 192, 256

    with torch.no_grad():
        for i in range(1):
            image = cv2.imread(cfg.input_image)

            # scale to fit canvas area
            canvas_size = [2346, 1440]
            fited_size = aspectFitSizeInSize([image.shape[1], image.shape[0]], canvas_size)
            fitted_input_image = cv2.resize(image, (fited_size[0], fited_size[1]), interpolation=cv2.INTER_CUBIC)

            #base 64 of original image
            retval, buffer = cv2.imencode('.jpg', fitted_input_image)
            input_image_base64 = base64.b64encode(buffer)

            #image = cv2.GaussianBlur(image, (7,7), 0)
            # the network is trained with 192*256 and the intrinsic parameter is set as ScanNet
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            image = cv2.GaussianBlur(image, (5,5), 0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            #
            
            image = transforms(image)
            image = image.to(device).unsqueeze(0)

            # forward pass
            logit, embedding, _, _, param = network(image)

            prob = torch.sigmoid(logit[0])
            
            # infer per pixel depth using per pixel plane parameter, currently Q_loss need a dummy gt_depth as input
            _, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, torch.ones_like(logit))

            # fast mean shift
            segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(prob, embedding[0], param, mask_threshold=0.1)

            # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
            # we thus use avg_pool_2d to smooth the segmentation results
            b = segmentation.t().view(1, -1, h, w)
            pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
            b = pooling_b.view(-1, h*w).t()
            segmentation = b

            # infer instance depth
            instance_loss, instance_depth, instance_abs_disntace, instance_parameter = \
                instance_parameter_loss(segmentation, sampled_segmentation, sample_param,
                                        torch.ones_like(logit), torch.ones_like(logit), False)

            # return cluster results
            predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

            # mask out non planar region
            predict_segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
            predict_segmentation = predict_segmentation.reshape(h, w)

            # change non planar to zero, so non planar region use the black color
            predict_segmentation += 1
            predict_segmentation[predict_segmentation==21] = 0

            # filter out the biggest plane
            uniquePlaneIds, uniquePlanePixelCounts = np.unique(predict_segmentation, return_counts=True)
            uniquePlaneIds = np.delete(uniquePlaneIds, 0)
            uniquePlanePixelCounts = np.delete(uniquePlanePixelCounts, 0)
            floor_plane_index = np.argmax(uniquePlanePixelCounts)
            floor_plane_id = uniquePlaneIds[floor_plane_index]

            predict_segmentation[predict_segmentation!=floor_plane_id] = 0
            predict_segmentation[predict_segmentation != 0] = 1

            # get full res contour and boundry
            floor_contour, floor_boundry_points = computeMainContourAndBoundry(predict_segmentation, (fited_size[0], fited_size[1]))
            
            
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

            #
            # calc pose
            #

            depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
            per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

            # use per pixel depth for non planar region
            depth = depth * (predict_segmentation != 0) + per_pixel_depth * (predict_segmentation == 0)

            viewspace_xyz = computeViewSpaceXYZ(depth, segmentation)
            low_contour, low_boundry_points = computeMainContourAndBoundry(predict_segmentation, (w, h))

            # compute the center of the contour
            M = cv2.moments(low_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            planep0 = viewspace_xyz[cY][cX]
            planep1 = viewspace_xyz[cY][cX + 1]
            planep2 = viewspace_xyz[cY + 1][cX]
            
            p0to1 = planep1 - planep0
            p0to1 = p0to1 / np.linalg.norm(p0to1)
            p0to2 = planep2 - planep0
            p0to2 = p0to2 / np.linalg.norm(p0to2)
            pnormal = -np.cross(p0to1, p0to2)

            floor_plane_normal = pnormal
            floor_viewspace_center = planep0

            up_vector = np.array([0,1,0])
            
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

            # build output filename
            extension = os.path.splitext(cfg.input_image)[1]
            basename = os.path.splitext(os.path.abspath(cfg.input_image))[0]

            output_json_file = os.path.join(basename + ".json" )

            output_json_stream = open(output_json_file, 'w')
            output_json_stream.write(json_template_string)
            output_json_stream.close()

            #cv2.imshow('image', image)
            #cv2.waitKey(0)


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    ex.add_config('./configs/config_unet_mean_shift.yaml')
    ex.run_commandline()

