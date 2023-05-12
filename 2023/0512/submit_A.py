from models import FaceModel
from datasets.dataset import FaceDataset, DataLoaderX
import sys
import glob
import torch
import os
import numpy as np
import cv2
import os.path as osp

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

resume_from = sys.argv[1]
submit_file = sys.argv[2]
model = FaceModel.load_from_checkpoint(resume_from).cuda()
model.eval()
outf = open(submit_file, 'w')
dev_set = FaceDataset(split='dev', return_path=True)
print('dev size:', len(dev_set))
for index, item in enumerate(dev_set):
    if index%200==0:
        print('processing', index)
    img, _, img_name = item
    img = torch.Tensor(img).unsqueeze(0).cuda()
    with torch.no_grad():
        pred = model(img).cpu().numpy()[0]
    pred = softmax(pred)
    score = pred[1]
    img_name = str(img_name)
    outf.write("%s %.5f\n"%(img_name, score))
test_set = FaceDataset(split='test', return_path=True)
print('test size:', len(test_set))
for index, item in enumerate(test_set):
    if index%200==0:
        print('processing', index)
    img, img_name = item
    img = torch.Tensor(img).unsqueeze(0).cuda()
    print(img.shape)
    with torch.no_grad():
        pred = model(img).cpu().numpy()[0]
    pred = softmax(pred)
    score = pred[1]
    img_name = str(img_name)
    outf.write("%s %.5f\n"%(img_name, score))

outf.close()


