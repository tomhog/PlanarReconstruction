import os
import os.path
import glob
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

from utils.vmutils import computeContoursAndBoundries as computeContoursAndBoundries
from utils.vmutils import computeViewSpaceXYZ as computeViewSpaceXYZ
from utils.vmutils import computeContourNormals as computeContourNormals
from utils.vmutils import vectorToAngles as vectorToAngles
from utils.vmutils import angleBetweenVectors as angleBetweenVectors
from utils.vmutils import pitchVector as pitchVector
from utils.vmutils import instance_ids as instance_ids
from utils.vmutils import generateVMEnvironmentTemplate as generateVMEnvironmentTemplate

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

    input_list = []
    if cfg.input_image == 'test':
        input_list = glob.glob("./tests/*.jpg")
    else:
        input_list.append(cfg.input_image)

    with torch.no_grad():
        for input_image_path in input_list:
            print('processing file: ' + str(input_image_path))

            image = cv2.imread(input_image_path)

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

            predict_segmentation[predict_segmentation > (len(instance_ids)-1)] = 0
            #predict_segmentation[predict_segmentation != 0] = 1

            # compute depth and viewspace xyz coords
            depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
            per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

            depth = depth * (predict_segmentation != 0) + per_pixel_depth * (predict_segmentation == 0)

            #generate vm data
            generateVMEnvironmentTemplate(predict_segmentation, segmentation, depth, (w, h), fited_size, input_image_path, input_image_base64, cfg)

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    ex.add_config('./configs/config_unet_mean_shift.yaml')
    ex.run_commandline()

