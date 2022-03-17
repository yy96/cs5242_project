import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)


def validation_result(validloader, model, device):
    target_ls = []
    preds_ls = []
    preds_threshold_ls = []
    with torch.no_grad():
        for data in validloader:
            ligand = data[0].to(device)
            protein = data[1].to(device)
            target = data[2].to(device).float()

            preds = model(protein.float(), ligand.float())

            preds_threshold = torch.where(
                preds <= torch.tensor(0.5).double(),
                torch.tensor(0).double(),
                torch.tensor(1).double(),
            )

            target_ls.append(target.numpy())
            preds_ls.append(preds.numpy())
            preds_threshold_ls.append(preds_threshold.numpy())

        final_target = np.concatenate(target_ls)
        final_preds = np.concatenate(preds_ls)
        final_preds_threshold = np.concatenate(preds_threshold_ls)

        try:
            accuracy = accuracy_score(final_target, final_preds_threshold)
        except:
            accuracy = -1

        try:
            f1 = f1_score(final_target, final_preds_threshold)
        except:
            f1 = -1

        try:
            precision = precision_score(final_target, final_preds_threshold)
        except:
            precision = -1
        try:
            recall = recall_score(final_target, final_preds_threshold)
        except:
            recall = -1
        try:
            auc_score = roc_auc_score(final_target, final_preds)
        except:
            auc_score = -1
        try:
            logloss_score = log_loss(final_target, final_preds)
        except:
            logloss_score = -1

        print(
            f"------ Accuracy: {accuracy}  F1: {f1}  precision: {precision}  recall:{recall}   auc:{auc_score}  logloss:{logloss_score}------"
        )

        return {
            "accuracy": accuracy,
            "F1": f1,
            "recall": recall,
            "auc": auc_score,
            "logloss": logloss_score,
        }
