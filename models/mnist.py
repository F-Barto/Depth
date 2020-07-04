import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import pytorch_lightning as pl



class SimpleClassifier(pl.LightningModule):
    def __init__(self, hparams, train_dataset, val_dataset, test_dataset):
        super(SimpleClassifier, self).__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.hparams = hparams

        # models images are (1, 28, 28) (channels, width, height)
        # just a very very simple model
        self.l1 = torch.nn.Linear(28 * 28, hparams.channels[0])
        self.l2 = torch.nn.Linear(hparams.channels[0], hparams.channels[1])
        self.l3 = torch.nn.Linear(hparams.channels[1], hparams.channels[2])

    def forward(self, x):
        x = torch.relu(self.l1(x.view(x.size(0), -1)))
        x = torch.relu(self.l2(x))
        return torch.relu(self.l3(x))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        wandb_logs = {'train_loss': loss}
        results = {
            'loss': loss,
            'log': wandb_logs,
            'progress_bar': wandb_logs
        }

        return results

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()

        wandb_logs = {'val_loss': val_loss_mean}
        results = {
            'val_loss_mean': val_loss_mean,
            'log': wandb_logs,
            'progress_bar': wandb_logs
        }

        return results

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y)
        return {'test_loss': test_loss}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()

        wandb_logs = {'test_loss': test_loss_mean}
        results = {
            'test_losss_mean': test_loss_mean,
            'log': wandb_logs,
            'progress_bar': wandb_logs
        }
        return results


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        # REQUIRED
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        # REQUIRED
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)