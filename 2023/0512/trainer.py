from argparse import ArgumentParser

import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import timm
from datasets.dataset import FaceDataset, DataLoaderX
from models import FaceModel





def cli_main():
    pl.seed_everything(727)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--backbone', default='resnet50d', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--root', default='data/SuHiFiMask/Challenge', type=str)
    parser.add_argument('--num-gpus', default=2, type=int)
    parser.add_argument('--tf32', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if not args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ------------
    # data
    # ------------
    train_set = FaceDataset(root_dir=args.root, split='train')
    val_set = FaceDataset(root_dir=args.root, split='dev')
    print('train data size:', len(train_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # ------------
    # model
    # ------------
    model = FaceModel(backbone=args.backbone, epoch=args.epoch)
    ckpt_path = 'work_dirs/1'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=ckpt_path,
            filename='{epoch:02d}-{val_loss:.6f}',
            save_top_k=10,
            mode='min',
            )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus = args.num_gpus,
        accelerator="ddp",
        benchmark=True,
		logger=TensorBoardLogger(osp.join(ckpt_path, 'logs')),
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=1,
        max_epochs=args.epoch,
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()

