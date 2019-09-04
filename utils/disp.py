"""
Copy from https://github.com/lxx1991/VS-ReID
"""

import os
import cv2
import numpy as np
import torchvision.transforms as transforms

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

colors = np.array([[0, 0, 0],
                   [0, 0, 255],
                   [0, 255, 0],
                   [255, 0, 0]],
                    dtype=np.uint8)

def scaleToAspectFitSize(rfit, rtarget):
    s = rtarget[0] / rfit[0]
    if (rfit[1] * s) <= rtarget[1]:
        return s
    return rtarget[1] / rfit[1]

def aspectFitSizeInSize(rfit, rtarget):
    s = scaleToAspectFitSize(rfit, rtarget)
    w = rfit[0] * s
    h = rfit[1] * s
    return [int(w),int(h)]

Tensor_to_Image = transforms.Compose([
    transforms.Normalize([0.0, 0.0, 0.0], [1.0/0.229, 1.0/0.224, 1.0/0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1.0, 1.0, 1.0]),
    transforms.ToPILImage()
])


def tensor_to_image(image):
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
