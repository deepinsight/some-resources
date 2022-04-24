import argparse

import cv2
import os
import os.path as osp
import numpy as np
import torch
import iresnet

from skimage import transform as trans
from scrfd import SCRFD
from utils import norm_crop

@torch.no_grad()
def inference(detector, net, img):
    bboxes, kpss = detector.detect(img, max_num=1)
    if bboxes.shape[0]==0:
        return None
    bbox = bboxes[0]
    kp = kpss[0]
    aimg = norm_crop(img, kp, image_size=112)

    aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
    aimg = np.transpose(aimg, (2, 0, 1))
    aimg = torch.from_numpy(aimg).unsqueeze(0).float()
    aimg.div_(255).sub_(0.5).div_(0.5)
    aimg = aimg.cuda()
    feat = net(aimg).cpu().numpy().flatten()
    feat /= np.sqrt(np.sum(np.square(feat)))
    return feat, bbox

if __name__ == "__main__":

    #init face detection
    detector = SCRFD(model_file = 'assets/det_10g.onnx')
    detector.prepare(0, det_thresh=0.5, input_size=(160, 160))

    #model-1
    net1 = iresnet.iresnet50()
    weight = 'assets/w600k_r50.pth'
    net1.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
    net1.eval().cuda()

    #model-2
    net2 = iresnet.iresnet100()
    weight = 'assets/glint360k_r100.pth'
    net2.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
    net2.eval().cuda()

    im = cv2.imread('your_photo.png')
    feat, bbox = inference(detector, net1, im)

