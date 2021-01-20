import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pl_bolts.datamodules import CIFAR10DataModule
from models import *



class ClassificationModel(pl.LightningModule) :
    def __init__(self, hparams: argparse.Namespace):     
        super().__init__()
        self.hparams = hparams
        self.model = DiracNet(3)
        self.xe = nn.CrossEntropyLoss()

    def loss_step(self, batch, batch_nb):
        out = self.model(batch['images'])
        cls_loss = self.xe(out[:,0].sigmoid(), mask)
        loss = cls_loss  
        return loss 
        
    def training_step(self, batch, batch_nb):
        loss = self.loss_step(batch, batch_nb)
        self.log('train_loss', loss.item())
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        loss = self.loss_step(batch, batch_nb)
        self.log('val_loss', loss.item())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return opt
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        #return [opt], [sch]

        





def main(data_dir, train_dir, batch_size=128, lr=1e-3, num_workers=1):
    params = argparse.Namespace(**locals())
    model = CornerDetectionModel(params)
    dm = Cifar10DataModule(params)


    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    ckcb = ModelCheckpoint(dirpath=ckpt_dir, filename='weights#{epoch}', save_top_k=None, period=1) 

    logger = TestTubeLogger(save_dir=os.path.join(train_dir, 'logs'), version=1)
    trainer = pl.Trainer(log_every_n_steps=10, checkpoint_callback=ckcb, logger=logger, gpus=1)
    trainer.fit(model, dm)



if __name__ == '__main__':
    import fire
    fire.Fire(main)
