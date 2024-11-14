# train.py

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ChessRecognitionDataset
from utils import download_chessred, extract_zipfile, recognition_accuracy

pl.seed_everything(42, workers=True)


class ChessDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for the chess recognition dataset."""

    def __init__(self, dataroot: str, batch_size: int, workers: int) -> None:
        super().__init__()
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.workers = workers

        # Data transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            ),
        ])

    def setup(self, stage: str = None) -> None:
        """Set up datasets for different stages."""
        if stage == 'fit' or stage is None:
            self.chess_train = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split='train',
                transform=self.transform
            )
            self.chess_val = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split='val',
                transform=self.transform
            )

        if stage == 'test' or stage == 'predict':
            self.chess_test = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split='test',
                transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_val,
            batch_size=self.batch_size,
            num_workers=self.workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_test,
            batch_size=self.batch_size,
            num_workers=self.workers
        )


class ChessMobileNet(pl.LightningModule):
    """MobileNetV2 model for chess recognition."""

    def __init__(self, lr: float = 1e-3, decay: int = 20) -> None:
        super().__init__()

        self.lr = lr
        self.decay = decay

        # Load MobileNetV2 backbone
        backbone = models.mobilenet_v2(weights="DEFAULT")
        num_filters = backbone.last_channel  # Usually 1280

        # Remove the classifier layer
        backbone.classifier = nn.Identity()

        self.feature_extractor = backbone

        num_target_classes = 64 * 13  # 64 squares * 13 classes

        # New classifier layer
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Reshape logits and labels for multi-label classification
        logits = logits.view(-1, 13)  # Each square has 13 classes
        labels = labels.view(-1, 13)

        return F.binary_cross_entropy_with_logits(logits, labels)

    def common_step(self, batch, batch_idx, stage: str):
        x, y = batch
        logits = self.forward(x)

        loss = self.cross_entropy_loss(logits, y)

        y_cat = torch.argmax(y.view(-1, 64, 13), dim=2)
        preds_cat = torch.argmax(logits.view(-1, 64, 13), dim=2)
        accuracy = recognition_accuracy(y_cat, preds_cat)

        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_acc', accuracy, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay, gamma=0.1)
        return [optimizer], [scheduler]


def main(args):
    data_module = ChessDataModule(
        dataroot=args.dataroot,
        batch_size=args.batch_size,
        workers=args.workers
    )

    model = ChessMobileNet(lr=args.lr, decay=args.decay)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_last=True,
        mode="min",
        save_top_k=args.topk,
        filename="mobilenet_{epoch:02d}-{val_loss:.4f}",
        save_weights_only=False
    )

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=args.ndevices,
        max_epochs=args.epochs,
        deterministic=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', required=True,
                        help="Path to the dataset.")

    parser.add_argument('--epochs', type=int, required=True,
                        help="Number of epochs to train the model.")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of samples per batch.")

    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Initial learning rate.")

    parser.add_argument('--decay', type=int, default=20,
                        help="Period (epochs) of learning rate decay by a factor of 10.")

    parser.add_argument('--topk', type=int, default=3,
                        help="Number k of top-performing model checkpoints to save.")

    parser.add_argument('--device', choices=["cpu", "gpu"], default="cpu",
                        help="Device to use for training ('cpu' or 'gpu').")

    parser.add_argument('--ndevices', type=int, default=1,
                        help="Number of devices to use for training.")

    parser.add_argument('--workers', type=int, default=4,
                        help="Number of workers to use for data loading.")

    parser.add_argument('--download', action="store_true",
                        help='Download the chess recognition dataset.')

    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    dataroot.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_chessred(dataroot)

        zip_path = dataroot / "images.zip"
        print()

        extract_zipfile(zip_file=zip_path, output_directory=dataroot)

        # Remove zip file
        zip_path.unlink(True)

    main(args)
