import sys
import os
import os.path as osp
import numpy as np
import datetime
import random
import torch
import glob
import time
import cv2
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import iresnet
from scrfd import SCRFD
from utils import norm_crop


class PyFAT:

    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = False
        self.num_iter = 100
        self.alpha = 1.0/255

    def set_cuda(self):
        self.is_cuda = True
        self.device = torch.device('cuda')
        torch.cuda.manual_seed_all(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def load(self, assets_path):
        detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
        ctx_id = -1 if not self.is_cuda else 0
        detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
        img_shape = (112,112)
        model = iresnet.iresnet50()
        weight = osp.join(assets_path, 'w600k_r50.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        
        # load face mask    
        mask_np = cv2.resize(cv2.imread(osp.join(assets_path, 'mask.png')), img_shape) / 255
        mask = torch.Tensor(mask_np.transpose(2, 0, 1)).unsqueeze(0)
        mask = F.interpolate(mask, img_shape).to(self.device)
        self.detector = detector
        self.model = model
        self.mask = mask

    def size(self):
        return 1

    def generate(self, im_a, im_v, n):
        h, w, c = im_a.shape
        assert len(im_a.shape) == 3
        assert len(im_v.shape) == 3
        bboxes, kpss = self.detector.detect(im_a, max_num=1)
        if bboxes.shape[0]==0:
            return None
        att_img, M = norm_crop(im_a, kpss[0], image_size=112)
        bboxes, kpss = self.detector.detect(im_v, max_num=1)
        if bboxes.shape[0]==0:
            return None
        vic_img, _ = norm_crop(im_v, kpss[0], image_size=112)

        att_img = att_img[:,:,::-1]
        vic_img = vic_img[:,:,::-1]
           
        # get victim feature
        vic_img = torch.Tensor(vic_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        vic_img.div_(255).sub_(0.5).div_(0.5)
        vic_feats = self.model.forward(vic_img)

        # process input
        att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        att_img.div_(255).sub_(0.5).div_(0.5)
        att_img_ = att_img.clone()
        att_img.requires_grad = True
        for i in tqdm(range(self.num_iter)):
            self.model.zero_grad()
            adv_images = att_img.clone() 
          
            # get adv feature
            adv_feats = self.model.forward(adv_images)

            # caculate loss and backward
            loss = torch.mean(torch.square(adv_feats - vic_feats))
            loss.backward(retain_graph=True)

            grad = att_img.grad.data.clone()
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = grad
            att_img.data = att_img.data - torch.sign(sum_grad) * self.alpha * (1 - self.mask)
            att_img.data = torch.clamp(att_img.data, -1.0, 1.0)
            att_img = att_img.data.requires_grad_(True) 
        # get diff and adv img
        diff = att_img - att_img_
        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
        diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
        diff_bgr = diff[:,:,::-1]
        adv_img = im_a + diff_bgr
        return adv_img

def main(args):

    # make directory
    save_dir = args.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tool = PyFAT()
    if args.device=='cuda':
        tool.set_cuda()
    tool.load('assets')
    
    for idname in range(1, 101):
        str_idname = "%03d"%idname
        iddir = osp.join('images', str_idname)
        att = osp.join(iddir, '0.png')
        vic = osp.join(iddir, '1.png')
        origin_att_img = cv2.imread(att)
        origin_vic_img = cv2.imread(vic)

        ta = datetime.datetime.now()
        adv_img = tool.generate(origin_att_img, origin_vic_img, 0)
        if adv_img is None:
            adv_img = origin_att_img
        tb = datetime.datetime.now()
        #print( (tb-ta).total_seconds() )
        save_name = '{}_2.png'.format(str_idname)
        cv2.imwrite(save_dir + '/' + save_name, adv_img)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='output directory', type=str, default='output/')
    parser.add_argument('--device', help='device to use', type=str, default='cpu')
    args = parser.parse_args()
    main(args)

