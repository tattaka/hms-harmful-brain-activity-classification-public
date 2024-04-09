import argparse
import datetime
import math
import os
import warnings
from typing import List

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from pytorch_lightning import LightningDataModule, callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from scipy.signal import butter, lfilter
from sklearn.model_selection import StratifiedGroupKFold
from timm.utils import ModelEmaV2
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "094"
COMMENT = """
baseline, 16ch,  label mixup, simplified head, sync_dist, many votes only, 
concat kaggle spectrogram w/ 16ch w/ preconv, fix preprocess, XYMask(separate),
fix df, first bn, eeg mixup only, tiling v2, refactor main.py, hflip, shuffle tiling before mixup,
vstack tiling, butter filter, more conv+bn, zoom aug, 2.5D(shuffle -> flip)
"""
BASE_DIR = "../../input/hms-harmful-brain-activity-classification/"
EEG_PATH = f"{BASE_DIR}/train_eegs/"
SPEC_PATH = f"{BASE_DIR}/train_spectrograms/"

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_signals(eeg_id, middle, eeg_path=EEG_PATH):

    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg_length = 50
    eeg = pd.read_parquet(f"{eeg_path}{eeg_id}.parquet")
    start = max(0, (middle - eeg_length // 2) * 200)
    end = min((middle + eeg_length // 2) * 200, len(eeg))
    eeg = eeg.iloc[start:end]
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            signals.append(x)
    signals_tmp = np.stack(signals)
    signals = np.zeros((len(signals), eeg_length * 200), dtype=np.float32)
    signals[:, : signals_tmp.shape[-1]] = signals_tmp
    return signals


def norm(x: torch.Tensor, start_dim=1, smooth=1e-5):
    dim = list(range(start_dim, x.ndim))
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + smooth)
    return x


class HMSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        overlap_df: pd.DataFrame = None,
        fmin: float = 0.5,
        fmax: float = 30,
        mode: str = "train",  # "train"  | "valid" | "test"
        spec_path: str = SPEC_PATH,
        eeg_path: str = EEG_PATH,
    ):
        self.mode = mode
        self.train = mode == "train" and overlap_df is not None
        self.df = df
        self.overlap_df = overlap_df
        self.fmin = fmin
        self.fmax = fmax
        self.spec_path = spec_path
        self.eeg_path = eeg_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        eeg_id = row.eeg_id
        if self.train:
            row2 = self.overlap_df[row.eeg_id == self.overlap_df.eeg_id]
            row = row2.iloc[np.random.randint(len(row2))]
        try:
            middle = int(row.eeg_label_offset_seconds) + 25
        except:  # for submission
            middle = 25
        signals = get_signals(eeg_id, middle, self.eeg_path)
        signals = np.stack(
            [
                butter_bandpass_filter(s, lowcut=self.fmin, highcut=self.fmax)
                for s in signals
            ]
        )
        signals = torch.tensor(signals, dtype=torch.float32)

        spectrogram = pd.read_parquet(f"{self.spec_path}{row.spectrogram_id}.parquet")
        try:
            spec_offset = int(row.spectrogram_label_offset_seconds)
            spectrogram = spectrogram.loc[
                (spectrogram.time >= spec_offset)
                & (spectrogram.time < spec_offset + 600)
            ]
        except:  # for submission
            spectrogram = spectrogram.iloc[0:300]
        spectrogram = spectrogram.iloc[:, 1:].to_numpy()
        if np.isnan(spectrogram).any():
            spectrogram[np.isnan(spectrogram)] = np.nanmean(spectrogram)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        specs = []
        for i in range(4):
            specs.append(spectrogram[:, i * 100 : (i + 1) * 100].T)
        # specs = norm(torch.tensor(np.stack(specs), dtype=torch.float32), start_dim=0)
        specs = torch.tensor(np.stack(specs), dtype=torch.float32)
        # specs = (specs + 40) / 40

        try:
            if self.train and np.random.rand() < 0.5:
                signals = signals.flip(-1)

            if self.train and np.random.rand() < 0.5:
                specs = specs.flip(-1)

            if self.train and np.random.rand() < 0.5:
                signals = signals.reshape(4, 4, -1).flip(0).reshape(16, -1)

            if self.train and np.random.rand() < 0.5:
                specs = specs.flip(0)

            labels = np.array(
                [
                    row.seizure_vote,
                    row.lpd_vote,
                    row.gpd_vote,
                    row.lrda_vote,
                    row.grda_vote,
                    row.other_vote,
                ],
                dtype=np.float32,
            )
            labels /= labels.sum()
            votes_type = np.array([row.num_votes > 7], dtype=np.float32)

            return {
                "eeg_id": eeg_id,
                "signals": signals,
                "labels": labels,
                "specs": specs,
                "votes_type": votes_type,
            }

        except:  # for submission
            return {
                "eeg_id": eeg_id,
                "signals": signals,
                "specs": specs,
            }


class HMSDataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        overlap_df: pd.DataFrame,
        fmin: float = 0.5,
        fmax: float = 30,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.overlap_df = overlap_df
        self._num_workers = num_workers
        self._batch_size = batch_size
        self.fmin = fmin
        self.fmax = fmax
        self.save_hyperparameters(
            "num_workers",
            "batch_size",
        )

    def create_dataset(self, mode: str = "train") -> HMSDataset:
        if mode == "train":
            return HMSDataset(
                df=self.train_df,
                overlap_df=self.overlap_df,
                fmin=self.fmin,
                fmax=self.fmax,
                mode="train",
            )
        else:
            return HMSDataset(
                df=self.valid_df,
                overlap_df=None,
                fmin=self.fmin,
                fmax=self.fmax,
                mode="valid",
            )

    def __dataloader(self, mode: str = "train") -> DataLoader:
        """Train/validation loaders."""
        dataset = self.create_dataset(mode)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=(mode == "train"),
            drop_last=(mode == "train"),
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="train")

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="valid")

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="test")

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("HMSDataModule")
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--batch_size",
            default=16,
            type=int,
            metavar="BS",
            help="number of sample in a batch",
            dest="batch_size",
        )
        return parent_parser


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x: torch.Tensor):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class AdaptiveConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1, 1)
        self.ap = nn.AdaptiveAvgPool3d(sz)
        self.mp = nn.AdaptiveMaxPool3d(sz)

    def forward(self, x: torch.Tensor):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def gem(x, p: float = 3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p: float = 3, eps: float = 1e-5, trainable: bool = True):
        super(GeM, self).__init__()
        self.trainable = trainable
        self.p = Parameter(torch.ones(1) * p) if self.trainable else p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


class XYMasking(nn.Module):
    def __init__(
        self,
        num_masks_x: int = 8,
        num_masks_y: int = 8,
        mask_x_length: float = 1 / 16,
        mask_y_length: float = 1 / 16,
        fill_value: float = 0.0,
        p: float = 0.0,
    ):
        super().__init__()
        self.num_masks_x = num_masks_x
        self.num_masks_y = num_masks_y
        self.mask_x_length = mask_x_length
        self.mask_y_length = mask_y_length
        self.fill_value = fill_value
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (bs, ch, h, w)
        bs, _, h, w = img.shape
        for i in range(bs):
            if self.training and np.random.rand() < self.p:
                x_masks = self.generate_mask(
                    np.random.randint(self.num_masks_x),
                    w,
                    h,
                    int(w * self.mask_x_length),
                    axis="x",
                )
                y_masks = self.generate_mask(
                    np.random.randint(self.num_masks_y),
                    w,
                    h,
                    int(h * self.mask_y_length),
                    axis="y",
                )
                masks = x_masks + y_masks
                # mask: (x1, y1, x2, y2)
                for mask in masks:
                    x1, y1, x2, y2 = mask
                    img[i, :, y1:y2, x1:x2] = self.fill_value
        return img

    def generate_mask(
        self, num_masks: int, width: int, height: int, max_length: int, axis: str
    ):
        masks = []
        for _ in range(num_masks):
            length = self.generate_mask_size(max_length)

            if axis == "x":
                x1 = np.random.randint(width - length + 1)
                y1 = 0
                x2, y2 = x1 + length, height
            else:  # axis == 'y'
                y1 = np.random.randint(height - length + 1)
                x1 = 0
                x2, y2 = width, y1 + length
            masks.append((x1, y1, x2, y2))

        return masks

    @staticmethod
    def generate_mask_size(mask_length: int) -> int:
        return np.random.randint(mask_length + 1)


def downsample_conv(
    in_channels: int,
    out_channels: int,
    stride: int = 2,
):
    return nn.Sequential(
        *[
            nn.Conv3d(
                in_channels,
                out_channels,
                1,
                stride=(1, stride, stride),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        ]
    )


class ResidualConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 2,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=(1, stride, stride),
                padding=1,
                bias=False,
                groups=mid_channels,
            ),
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.act3 = nn.GELU()
        self.downsample = downsample_conv(
            in_channels,
            out_channels,
            stride=stride,
        )
        self.stride = stride
        self.zero_init_last()

    def zero_init_last(self):
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


class HMSModel2_5D(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        specaug_p: float = 0.5,
        in_chans: int = 16,
        num_class: int = 6,
        img_size: List[int] = None,
    ):
        super().__init__()
        self.specaug_eeg = XYMasking(p=specaug_p)
        self.specaug_kaggle = XYMasking(p=specaug_p)

        self.kaggle_stem = nn.Sequential(
            nn.Conv2d(4, in_chans, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_chans),
            nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(in_chans),
        )
        self.eeg_stem = nn.BatchNorm2d(in_chans)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=4,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            img_size=(
                img_size
                if ("swin" in model_name)
                or ("coat" in model_name)
                or ("max" in model_name)
                else None
            ),
        )
        try:
            num_features = self.encoder.num_features
        except:
            num_features = self.encoder.embed_dim
        self.conv3d = nn.Sequential(
            *[
                ResidualConv3D(
                    num_features,
                    num_features // 4,
                    num_features,
                    1,
                )
                for _ in range(3)
            ]
        )
        self.model_name = model_name
        self.output_fmt = getattr(self.encoder, "output_fmt", "NHCW")
        self.head = nn.Sequential(
            nn.Conv3d(num_features, 512, kernel_size=1),
            AdaptiveConcatPool3d(1),
            Flatten(),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(1024, num_class),
        )

    def forward_image_feats(self, img, specs):
        # img -> (bs, in_chans, h, w)
        specs = self.kaggle_stem(specs)
        specs = F.interpolate(
            specs, size=(img.shape[-2], img.shape[-1]), mode="nearest"
        )
        img = self.eeg_stem(img)
        bs, _, h, w = img.shape
        img = img.reshape(bs * 4, 4, h, w)
        specs = specs.reshape(bs * 4, 4, h, w)
        img = self.specaug_eeg(img)
        specs = self.specaug_kaggle(specs)

        img = torch.concat([img, specs], -2)  # (bs * zdim, 4, h * 2, w)
        img_feat = self.encoder.forward_features(img)
        if self.output_fmt == "NHWC":
            img_feat = img_feat.permute(0, 3, 1, 2).contiguous()

        _, ch, h, w = img_feat.shape
        img_feat = img_feat.reshape(bs, 4, ch, h, w).transpose(
            1, 2
        )  # (bs, ch, 4, h, w)
        img_feat = self.conv3d(img_feat)  # (bs, ch, 4, h, w) -> (bs, ch, 4, h, w)
        return img_feat

    def forward(
        self,
        img: torch.Tensor,
        specs: torch.Tensor,
    ):
        """
        img: (bs, ch, h, w)
        """
        img_feats = self.forward_image_feats(img, specs)
        return self.forward_head(img_feats)

    def forward_head(self, img_feats):
        output = self.head(img_feats)
        return output

    def set_grad_checkpointing(self, enable: bool = True):
        self.encoder.set_grad_checkpointing(enable)


class Mixup(object):
    def __init__(self, p: float = 0.5, alpha: float = 0.5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self):
        self.lam = 1.0


class HMSPipeline(torch.nn.Module):
    def __init__(
        self,
        height: int = 128,
        width: int = 256,
        eeg_length: int = 50,
        win_length: int = 128,
        fmin: float = 0.5,
        fmax: float = 30,
    ):
        super().__init__()
        n_fft = 1024 * height // 128
        self.height = height
        self.width = width
        self.eeg_length = eeg_length
        self.mel_spec = MelSpectrogram(
            sample_rate=200,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=eeg_length * 200 // width,
            n_mels=height,
            f_min=fmin,
            f_max=fmax,
        )
        # self.mel_spec.spectrogram.window = nn.Parameter(self.mel_spec.spectrogram.window)
        # self.mel_spec.mel_scale.fb = nn.Parameter(self.mel_spec.mel_scale.fb)
        self.amplitude_to_db = AmplitudeToDB()

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        middle = 25
        eeg_length = self.eeg_length

        # zoom augment
        if self.training and np.random.rand() < 0.5:
            eeg_length = min(
                self.eeg_length
                + np.random.randint(-self.eeg_length // 4, self.eeg_length // 4),
                50,
            )
        self.mel_spec.spectrogram.hop_length = eeg_length * 200 // self.width
        self.mel_spec.hop_length = eeg_length * 200 // self.width

        signals = signals[
            :,
            :,
            (middle - eeg_length // 2) * 200 : (middle + eeg_length // 2) * 200,
        ]
        # signals -> (bs, 16, length)
        bs, ch, length = signals.shape
        signals = signals.reshape((bs * ch, -1))
        # Convert to melspectrogram
        mel = self.mel_spec(signals)

        # LOG TRANSFORM
        mel_db = self.amplitude_to_db(mel)[..., : self.height, : self.width]
        mel_db = F.pad(
            mel_db,
            (0, self.width - mel_db.shape[-1], 0, self.height - mel_db.shape[-2]),
        )
        assert mel_db.shape[-1] == self.width
        assert mel_db.shape[-2] == self.height
        # mel_db = power_to_db(mel)[..., : self.height, : self.width]
        width = mel_db.shape[2]
        height = mel_db.shape[1]
        mel_db = mel_db.reshape((bs, ch, height, width))
        # mel_db = mel_db.reshape(bs, 4, 4, height, width).mean(1)
        # mel_db = norm(mel_db)
        # mel_db = (mel_db + 40) / 40
        return mel_db


class HMSLightningModel2_5D(pl.LightningModule):
    def __init__(
        self,
        height: int = 128,
        width: int = 256,
        eeg_length: int = 50,
        win_length: int = 128,
        fmin: float = 0.5,
        fmax: float = 30,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        specaug_p: float = 0.5,
        in_chans: int = 16,
        num_class: int = 6,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.5,
        no_mixup_epochs: int = 0,
        lr: float = 1e-3,
        backbone_lr: float = None,
        weight_decay: float = 0.0001,
        enable_gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.fmin = fmin
        self.fmax = fmax
        self.pipeline = HMSPipeline(
            height=height,
            width=width,
            eeg_length=eeg_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
        )
        self.backbone_lr = backbone_lr if backbone_lr is not None else lr
        self.weight_decay = weight_decay
        self.__build_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            specaug_p=specaug_p,
            in_chans=in_chans,
            num_class=num_class,
            img_size=[height * 2, width],
        )
        if enable_gradient_checkpoint:
            self.model.set_grad_checkpointing(enable_gradient_checkpoint)
        self.mixupper = Mixup(p=mixup_p, alpha=mixup_alpha)
        self.mixup_alpha = mixup_alpha
        self.no_mixup_epochs = no_mixup_epochs
        self.save_hyperparameters()

    def __build_model(
        self,
        model_name: str = "resnet34",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        specaug_p: float = 0.5,
        in_chans: int = 16,
        num_class: int = 6,
        img_size: List[int] = None,
    ):
        self.model = HMSModel2_5D(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            specaug_p=specaug_p,
            in_chans=in_chans,
            num_class=num_class,
            img_size=img_size,
        )
        self.model_ema = ModelEmaV2(self.model, decay=0.998)
        self.criterions = {
            "kldiv": nn.KLDivLoss(reduction="batchmean"),
        }

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        # smooth = 0.1
        # true = labels["targets"] * (1 - (smooth / 0.5)) + smooth

        losses["kldiv"] = self.criterions["kldiv"](
            F.log_softmax(outputs["logits"], dim=1),
            labels["targets"].to(dtype=outputs["logits"].dtype),
        )
        losses["loss"] = losses["kldiv"]
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        self.mixupper.init_lambda()
        signals, labels, specs = batch["signals"], batch["labels"], batch["specs"]
        if (
            self.mixupper.do_mixup
            and self.current_epoch < self.trainer.max_epochs - self.no_mixup_epochs
        ):
            signals = self.mixupper.lam * signals + (
                1 - self.mixupper.lam
            ) * signals.flip(0)
            specs = self.mixupper.lam * specs + (1 - self.mixupper.lam) * specs.flip(0)
            labels = self.mixupper.lam * labels + (1 - self.mixupper.lam) * labels.flip(
                0
            )
        else:
            pass
        image = self.pipeline(signals)
        outputs["logits"] = self.model(image, specs)

        loss_target["targets"] = labels
        losses = self.calc_loss(outputs, loss_target)

        step_output.update(losses)
        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_kldiv_loss=losses["kldiv"],
            ),
            sync_dist=True,
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}

        signals, labels, specs = batch["signals"], batch["labels"], batch["specs"]
        image = self.pipeline(signals)
        outputs["logits"] = self.model_ema.module(image, specs)

        loss_target["targets"] = labels
        losses = self.calc_loss(outputs, loss_target)

        step_output.update(losses)

        self.log_dict(
            dict(
                val_loss=losses["loss"],
                val_kldiv_loss=losses["kldiv"],
            ),
            sync_dist=True,
        )
        return step_output

    def on_validation_epoch_end(self):
        pass

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
                "lr": self.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.head.named_parameters())
                    + list(self.model.kaggle_stem.named_parameters())
                    + list(self.model.eeg_stem.named_parameters())
                    + list(self.model.conv3d.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.head.named_parameters())
                    + list(self.model.kaggle_stem.named_parameters())
                    + list(self.model.eeg_stem.named_parameters())
                    + list(self.model.conv3d.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
                "lr": self.lr,
            },
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        optimizer = AdamW(self.get_optimizer_parameters(), eps=1e-6)
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil((max_train_steps * 2) / 100) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("HMSLightningModel2_5D")
        parser.add_argument(
            "--height",
            default=128,
            type=int,
            metavar="H",
            dest="height",
        )
        parser.add_argument(
            "--width",
            default=256,
            type=int,
            metavar="W",
            dest="width",
        )
        parser.add_argument(
            "--win_length",
            default=128,
            type=int,
            metavar="WL",
            dest="win_length",
        )
        parser.add_argument(
            "--eeg_length",
            default=50,
            type=int,
            metavar="EL",
            help="eeg signal length(sec)",
            dest="eeg_length",
        )
        parser.add_argument(
            "--model_name",
            default="resnet34",
            type=str,
            metavar="MN",
            help="Name (as in ``timm``) of the feature extractor",
            dest="model_name",
        )
        parser.add_argument(
            "--drop_path_rate",
            default=None,
            type=float,
            metavar="DPR",
            dest="drop_path_rate",
        )
        parser.add_argument(
            "--specaug_p",
            default=0.5,
            type=float,
            metavar="SP",
            dest="specaug_p",
        )
        parser.add_argument(
            "--in_chans",
            default=16,
            type=int,
            metavar="ICH",
            dest="in_chans",
        )
        parser.add_argument(
            "--num_class",
            default=6,
            type=int,
            metavar="OCL",
            dest="num_class",
        )
        parser.add_argument(
            "--mixup_p", default=0.0, type=float, metavar="MP", dest="mixup_p"
        )
        parser.add_argument(
            "--mixup_alpha", default=0.0, type=float, metavar="MA", dest="mixup_alpha"
        )
        parser.add_argument(
            "--no_mixup_epochs",
            default=0,
            type=int,
            metavar="NME",
            dest="no_mixup_epochs",
        )
        parser.add_argument(
            "--lr",
            default=1e-3,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--backbone_lr",
            default=None,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="backbone_lr",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0001,
            type=float,
            metavar="WD",
            help="initial weight decay",
            dest="weight_decay",
        )
        parent_parser.add_argument(
            "--enable_gradient_checkpoint",
            action="store_true",
            help="enable set_gradient_checkpoint",
            dest="enable_gradient_checkpoint",
        )
        return parent_parser


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parent_parser.add_argument(
        "--fmin",
        default=0.5,
        type=float,
        metavar="FMin",
        dest="fmin",
    )
    parent_parser.add_argument(
        "--fmax",
        default=30,
        type=float,
        metavar="FMax",
        dest="fmax",
    )
    parent_parser.add_argument(
        "--gpus", type=int, default=4, help="number of gpus to use"
    )
    parent_parser.add_argument(
        "--epochs", default=10, type=int, metavar="N", help="total number of epochs"
    )
    parent_parser.add_argument(
        "--precision",
        # default="16-mixed",
        default="32",
    )
    parser = HMSLightningModel2_5D.add_model_specific_args(parent_parser)
    parser = HMSDataModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    assert args.fold < 5
    overlap_df = pd.read_csv(f"{BASE_DIR}/train.csv")
    overlap_df["num_votes"] = overlap_df.values[:, -6:].sum(-1)
    overlap_df = overlap_df[overlap_df["num_votes"] > 7].reset_index(drop=True)
    df = overlap_df.groupby("eeg_id").first().reset_index()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(
        sgkf.split(df, df["expert_consensus"], df["patient_id"])
    ):
        if fold != args.fold:
            continue
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        datamodule = HMSDataModule(
            train_df=train_df,
            valid_df=valid_df,
            overlap_df=overlap_df,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        model = HMSLightningModel2_5D(
            height=args.height,
            width=args.width,
            eeg_length=args.eeg_length,
            win_length=args.win_length,
            fmin=args.fmin,
            fmax=args.fmax,
            model_name=args.model_name,
            pretrained=True,
            drop_path_rate=args.drop_path_rate,
            specaug_p=args.specaug_p,
            in_chans=args.in_chans,
            num_class=args.num_class,
            mixup_p=args.mixup_p,
            mixup_alpha=args.mixup_alpha,
            no_mixup_epochs=args.no_mixup_epochs,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            enable_gradient_checkpoint=args.enable_gradient_checkpoint,
        )

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{fold}"
        print(f"logdir = {logdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            mode="min",
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        if not args.debug:
            wandb_logger = WandbLogger(
                name=f"exp{EXP_ID}/{args.logdir}/fold{fold}",
                tags=[f"fold{fold}"],
                save_dir=logdir,
                project="hms-harmful-brain-activity-classification",
                notes=COMMENT,
            )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, log_rank_zero_only=True
        )
        trainer = pl.Trainer(
            default_root_dir=logdir,
            sync_batchnorm=True,
            gradient_clip_val=1.0,
            precision=args.precision,
            devices=args.gpus,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            # strategy="ddp",
            max_epochs=args.epochs,
            logger=wandb_logger if not args.debug else True,
            callbacks=[
                loss_checkpoint,
                lr_monitor,
                early_stopping,
            ],
            fast_dev_run=args.debug,
            num_sanity_val_steps=0,
            accumulate_grad_batches=max(16 // args.batch_size, 1),
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
