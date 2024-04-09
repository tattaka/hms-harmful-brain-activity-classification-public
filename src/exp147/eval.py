import argparse
import datetime
import warnings
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from main import BASE_DIR, EXP_ID, HMSDataModule, HMSLightningModel
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn import functional as F
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser = HMSLightningModel.add_model_specific_args(parent_parser)
    parser = HMSDataModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    assert args.fold < 5
    overlap_df = pd.read_csv(f"{BASE_DIR}/train.csv")
    overlap_df["num_votes"] = overlap_df.values[:, -6:].sum(-1)
    # patient_cv = pd.read_csv("../../input/patient_fold.csv")
    # patient_cv = patient_cv.groupby("patient_id").first().reset_index()
    # df = pd.merge(overlap_df, patient_cv, on="patient_id")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    overlap_df["fold"] = -1
    for fold, (train_idx, valid_idx) in enumerate(
        sgkf.split(overlap_df, overlap_df["expert_consensus"], overlap_df["patient_id"])
    ):
        overlap_df.loc[valid_idx, "fold"] = fold
    df = overlap_df.copy()
    preds_all = []
    logits_all = []
    labels_all = []
    eeg_ids_all = []
    target_columns = [c for c in df.columns if "_vote" in c and c != "num_votes"]
    for i, c in enumerate(target_columns):
        df[f"oof_{c}"] = None
        df[f"oof_logits_{c}"] = None
    df["kl_div"] = None
    for fold in range(5):
        train_idx = df[df.fold != fold].index.values
        valid_idx = df[df.fold == fold].index.values
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        dataloader = HMSDataModule(
            train_df=train_df,
            valid_df=valid_df,
            overlap_df=overlap_df,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        ).val_dataloader()

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{fold}"
        print(f"logdir: {logdir}")
        model = (
            HMSLightningModel.load_from_checkpoint(
                glob(f"{logdir}/**/best_loss.ckpt", recursive=True)[0],
                pretrained=False,
            )
            .eval()
            .to(device=device)
        )
        preds_fold = []
        logits_fold = []
        labels_fold = []
        eeg_ids_fold = []
        for batch in tqdm(dataloader):
            eeg_ids, signals, labels, specs = (
                batch["eeg_id"],
                batch["signals"],
                batch["labels"],
                batch["specs"],
            )
            eeg_ids_fold.append(eeg_ids)
            image = model.pipeline(signals.to(device=device))
            specs = specs.to(device=device)
            with torch.no_grad():
                logit = model.model_ema.module(image, specs)
                pred = torch.softmax(logit, dim=1).detach().cpu().numpy()
                preds_fold.append(pred)
                logits_fold.append(logit.detach().cpu().numpy())
            labels_fold.append(labels.numpy())
        preds_fold = np.concatenate(preds_fold)
        logits_fold = np.concatenate(logits_fold)
        labels_fold = np.concatenate(labels_fold)
        eeg_ids_fold = np.concatenate(eeg_ids_fold)
        preds_all.append(preds_fold)
        logits_all.append(logits_fold)
        labels_all.append(labels_fold)
        eeg_ids_all.append(eeg_ids_fold)
        scores = np.asarray(
            [
                F.kl_div(
                    torch.log(torch.tensor(pred[None, ...])),
                    torch.tensor(label[None, ...]),
                    reduction="batchmean",
                ).item()
                for pred, label in zip(preds_fold, labels_fold)
            ]
        )
        for i, c in enumerate(target_columns):
            df.loc[valid_idx, f"oof_{c}"] = preds_fold[:, i]
            df.loc[valid_idx, f"oof_logits_{c}"] = logits_fold[:, i]
        df.loc[valid_idx, "kl_div"] = scores
    preds_all = np.concatenate(preds_all)
    logits_all = np.concatenate(logits_all)
    labels_all = np.concatenate(labels_all)
    eeg_ids_all = np.concatenate(eeg_ids_all)
    for i in range(5):
        print(
            f"CV fold{i}:",
            df[(df.num_votes > 7) & (df.fold == i)]
            .groupby("eeg_id")
            .first()
            .kl_div.mean(),
        )
    print(
        "CV mean:",
        df[(df.num_votes > 7)].groupby("eeg_id").first().kl_div.mean(),
    )
    df.to_csv(f"../../logs/exp{EXP_ID}/{args.logdir}/result_df.csv", index=False)


if __name__ == "__main__":
    main(get_args())
