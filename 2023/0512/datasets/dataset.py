import os
import os.path as osp
import queue as Queue
import mxnet as mx
import pickle
import threading
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
#from .augs import RectangleBorderAugmentation

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch



class FaceDataset(Dataset):
    def __init__(self, root_dir='data/SuHiFiMask/Challenge', split='train', return_path=False):
        super(FaceDataset, self).__init__()

        #self.local_rank = local_rank
        #self.is_train = is_train
        self.input_size = 224
        #self.num_kps = 68
        transform_list = []
        if split=='train':
            transform_list += \
                [
                    A.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
                    A.ToGray(p=0.1),
                    A.ISONoise(p=0.1),
                    A.MedianBlur(blur_limit=(1,7), p=0.1),
                    A.GaussianBlur(blur_limit=(1,7), p=0.1),
                    A.MotionBlur(blur_limit=(5,12), p=0.1),
                    A.ImageCompression(quality_lower=50, quality_upper=90, p=0.05),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, interpolation=cv2.INTER_LINEAR, 
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.8),
                    A.HorizontalFlip(p=0.5),
                    #RectangleBorderAugmentation(limit=0.2, fill_value=0, p=0.1),
                ]
        transform_list += \
            [
                A.geometric.resize.Resize(self.input_size, self.input_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        self.transform = A.Compose(
            transform_list,
        )
        self.split = split
        self.data_list = []
        self.label_list = []
        if split=='train':
            self.img_root = osp.join(root_dir, 'phase1')
            self.data_list = []
            for line in open(osp.join(self.img_root, 'train_label.txt'), 'r'):
                line = line.strip().split()
                assert len(line)==2
                label = int(line[1])
                self.data_list.append(line[0])
                self.label_list.append(label)
        elif split=='dev':
            self.img_root = osp.join(root_dir, 'phase1')
            img_root2 = osp.join(root_dir, 'phase2')
            self.data_list = []
            for line in open(osp.join(img_root2, 'dev_label.txt'), 'r'):
                line = line.strip().split()
                assert len(line)==2
                label = int(line[1])
                self.data_list.append(line[0])
                self.label_list.append(label)
        elif split=='test':
            self.img_root = osp.join(root_dir, 'phase2')
            self.data_list = []
            for line in open(osp.join(self.img_root, 'test.txt'), 'r'):
                line = line.strip().split()
                assert len(line)==1
                self.data_list.append(line[0])
        self.data_list = np.array(self.data_list)
        self.label_list = np.array(self.label_list)
        self.return_path = return_path

        logging.info('len:%d'%len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_name = self.data_list[index]
        img_path = osp.join(self.img_root, img_name)
        img = cv2.imread(img_path)[:,:,::-1]
        if self.transform is not None:
            t = self.transform(image=img)
            img = t['image']
        if self.split!='test':
            label = self.label_list[index]
            label = torch.tensor(label, dtype=torch.int64)
            if not self.return_path:
                return img, label
            else:
                return img, label, img_name
        else:
            if not self.return_path:
                return img
            else:
                return img, img_name

