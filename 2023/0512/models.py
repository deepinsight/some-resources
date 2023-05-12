import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import timm

class FaceModel(pl.LightningModule):
    def __init__(self, backbone, epoch):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model(backbone, num_classes=2, pretrained=True)
        self.epoch = epoch
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # use forward for inference/predictions
        y = self.backbone(x)
        return y

    def cal_loss(self, y_hat, y):
        loss = self.loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=0.0002)
        opt = torch.optim.SGD(self.parameters(), lr = 0.1, momentum=0.9, weight_decay = 0.0005)
        epoch_steps = [int(self.epoch*0.3), int(self.epoch*0.6), int(self.epoch*0.9)]
        print('epoch_steps:', epoch_steps)
        def lr_step_func(epoch):
            return 0.1 ** len([m for m in epoch_steps if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt, lr_lambda=lr_step_func)
        lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'epoch',
                'frequency': 1}
        return [opt], [lr_scheduler]
