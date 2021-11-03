from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
from pathlib import Path
import random
from loguru import logger
import argparse
import pathlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy

import torchvision.transforms as transforms
from torchvision.transforms import (
    Compose,
    Lambda,
)

from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
)


class x3d_tfms:
    def __init__(self, backbone, resize=False):
        self.backbone = backbone
        self.resize = resize
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.frames_per_second = 30
        model_transform_params  = {
            "x3d_xs": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            }
        }

        # Get transform parameters based on model
        self.transform_params = model_transform_params[backbone]

    def get_tfms(self):
        tfms_list = [
                    UniformTemporalSubsample(self.transform_params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    Normalize(self.mean, self.std),
                ]
        if self.resize:
            tfms_list += [ShortSideScale(size=self.transform_params["side_size"]),
                          CenterCropVideo(crop_size=(self.transform_params["crop_size"], 
                                                     self.transform_params["crop_size"]))
                          ]

        # Note that this transform is specific to the x3d model.
        tfms =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                tfms_list
            ),
        )

        # The duration of the input clip is also specific to the model.
        clip_duration = (self.transform_params["num_frames"] * self.transform_params["sampling_rate"])/self.frames_per_second

        return tfms, clip_duration


class DeepFakeDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_path: Union[str, Path], 
        backbone: str = "x3d_s",
        num_workers: int = 8, 
        batch_size: int = 32,
        resize: bool = False):
        """DeepFakeDataModule.

        Args:
            data_path: folder where train.csv and valid.csv are located
            num_workers: number of CPU workers
            batch_size: number of sample in a batch
            clip_duration: duration of sampled clip for each video in seconds
        """
        super().__init__()

        self.data_path = data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        if "x3d" in backbone:
            x3d = x3d_tfms(backbone, resize)
            self.tfms, self.clip_duration = x3d.get_tfms()
        else:
            raise ValueError(f"Only x3d backbones supported. {backbone} provided instead.")

    @property
    def train_transform(self):
        return self.tfms

    @property
    def valid_transform(self):
        return self.tfms

    def create_dataset(
        self,
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_path_prefix: str = "",
        decode_audio: bool = False,
        decoder: str = "pyav",):
        return labeled_video_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            video_sampler=video_sampler,
            transform=transform,
            video_path_prefix=video_path_prefix,
            decode_audio=decode_audio,
            decoder=decoder,)

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        if train:
            dataset = self.create_dataset(data_path=self.data_path+"/train_faces.csv", 
                                          clip_sampler=make_clip_sampler("random", self.clip_duration),
                                          transform=self.train_transform)
        else:
            dataset = self.create_dataset(data_path=self.data_path+"/valid_faces.csv", 
                                          clip_sampler=make_clip_sampler("uniform", self.clip_duration),
                                          transform=self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        logger.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        logger.info("Validation data loaded.")
        return self.__dataloader(train=False)


class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestone: int = 5, unfreeze_top_layers: int = 3, train_bn: bool = False):
        super().__init__()
        self.milestone = milestone
        self.train_bn = train_bn
        self.unfreeze_top_layers = unfreeze_top_layers

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        opt_pgs = optimizer.param_groups
        pl_module.log("lr head param group", opt_pgs[0]["lr"])
        pl_module.log("# param groups", len(opt_pgs))

        if epoch == self.milestone:
            # unfreeze `unfreeze_top_layers` last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-self.unfreeze_top_layers:], optimizer=optimizer, train_bn=self.train_bn
            )


class DeepFakeModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "x3d_s",
        train_bn: bool = False,
        milestone: int = 5,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        **kwargs,
    ) -> None:
        """TransferLearningModel.

        Args:
            backbone: Name of the feature extractor in Torch Hub
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.milestone = milestone

        self.__build_model()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.soft = nn.Softmax()
        self.save_hyperparameters()

    def __build_model(self):
        """Define model layers and loss."""
        # 1. Load pre-trained network:
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=self.backbone, pretrained=True)
        layers = list(model.blocks.children())
        _layers = layers[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        # 2. Classifier:
        self.fc = layers[-1]
        self.fc.proj = nn.Linear(in_features=2048, out_features=2, bias=True)

        # 3. Loss:
        self.loss_func = F.cross_entropy

    def forward(self, x):
        """Forward pass.

        Returns logits.
        """

        # 1. Feature extraction:
        x = self.feature_extractor(x)

        # 2. Classifier (returns logits):
        x = self.fc(x)

        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        y_hat = self.forward(batch["video"])

        # 2. Compute loss
        train_loss = self.loss(y_hat, batch["label"])

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(self.soft(y_hat), batch["label"]), prog_bar=True)
        self.log("train_loss", train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        y_hat = self.forward(batch["video"])

        # 2. Compute loss
        self.log("val_loss", self.loss(y_hat, batch["label"]), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(self.soft(y_hat), batch["label"]), prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=[self.milestone], gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]