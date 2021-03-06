import os
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 
import pytorch_lightning as pl
import torchvision

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datamodules import CIFAR10DataModule
from models import *



class ClassificationModel(pl.LightningModule) :
    def __init__(self, hparams: argparse.Namespace):     
        super().__init__()
        self.hparams = hparams
        self.model = create_RepVGG_A0(num_classes=10)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log('Training Loss', loss)
        return loss

    def _evaluate(self, batch, batch_idx, stage=None):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

        return loss, acc

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, batch_idx, 'val')[0]

    def test_step(self, batch, batch_idx):
        loss, acc = self._evaluate(batch, batch_idx, 'test')
        self.log_dict({'test_loss': loss, 'test_acc': acc})

    def configure_optimizers(self):
        #opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        #return opt
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=math.ceil(45000 / self.hparams.batch_size)),
                'interval': 'step',
            }
        }


def show(dm):
    from torchvision.utils import make_grid
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
    import cv2

    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]]
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    unnormalize = lambda x: 255* ((x*std)+mean)
    for x,y in dm.train_dataloader():
        im = make_grid(x.cpu()).permute(1,2,0).contiguous().numpy()
        im = unnormalize(im).astype(np.uint8)
        cv2.imshow('im', im[...,::-1])
        cv2.waitKey()



def main(train_dir, batch_size=128, lr=1e-3, num_workers=1):
    params = argparse.Namespace(**locals())
    model = ClassificationModel(params)
    dm = CIFAR10DataModule()

    normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
    dm.train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    dm.test_transforms = dm.val_transformrs = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    # show(dm)


    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    ckcb = ModelCheckpoint(dirpath=ckpt_dir, filename='weights#{epoch}', save_top_k=None, period=1) 

    logger = TestTubeLogger(save_dir=os.path.join(train_dir, 'logs'), version=1)
    trainer = pl.Trainer(log_every_n_steps=10, checkpoint_callback=ckcb, logger=logger, gpus=1)
    trainer.fit(model, dm)



if __name__ == '__main__':
    import fire
    fire.Fire(main)
