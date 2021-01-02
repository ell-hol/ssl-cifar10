import torch

from torch.nn.functional import cross_entropy
import pytorch_lightning as pl
from torch.optim import optimizer
from torchvision import models
from torch.optim import Adam

from pytorch_lightning.metrics.functional import accuracy 

from pl_bolts.datamodules import CIFAR10DataModule

from pl_bolts.models.self_supervised import SwAV

pl.seed_everything(42)

weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
swav = SwAV.load_from_checkpoint(weight_path, strict=True)

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = swav.model
        self.finetune_layer = torch.nn.Linear(3000, num_classes)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.trainer.current_epoch < 10:
            with torch.no_grad():
                (f1,f2) = self.backbone(x)
                features = f2
        else:
            (f1,f2) = self.backbone(x)
            features = f2

        preds = self.finetune_layer(features)
        acc = accuracy(preds, y)
        loss = cross_entropy(preds, y)
        self.log("train_accuracy", acc)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        (f1,f2) = self.backbone(x)
        features = f2
        preds = self.finetune_layer(features)
        acc = accuracy(preds, y)
        loss = cross_entropy(preds, y)
        self.log("val_accuracy", acc)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        # optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
        return optimizer


if __name__ == "__main__":
    dm = CIFAR10DataModule('.')
    model = ImageClassifier()
    trainer = pl.Trainer(progress_bar_refresh_rate=20, gpus=1, auto_scale_batch_size="power")
    trainer.fit(model, dm)
