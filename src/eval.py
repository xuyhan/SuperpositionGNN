import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class MultiLabelEvaluator:

    def __init__(self, k: int, threshold: float = 0.5, device: str = "cpu", eps: float = 1e-12):
        self.k = k
        self.th = threshold
        self.eps = eps

        self.tp = torch.zeros(k, dtype=torch.long, device=device)
        self.fp = torch.zeros(k, dtype=torch.long, device=device)
        self.tn = torch.zeros(k, dtype=torch.long, device=device)
        self.fn = torch.zeros(k, dtype=torch.long, device=device)

        self._all_targets = []
        self._all_scores = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        preds = (probs >= self.th).long()
        targets = targets.long()

        self.tp += ((preds == 1) & (targets == 1)).sum(dim=0)
        self.fp += ((preds == 1) & (targets == 0)).sum(dim=0)
        self.tn += ((preds == 0) & (targets == 0)).sum(dim=0)
        self.fn += ((preds == 0) & (targets == 1)).sum(dim=0)

        self._all_targets.append(targets.detach().cpu())
        self._all_scores.append(probs.detach().cpu())

    def compute(self):
        tp, fp, tn, fn = (x.float() for x in (self.tp, self.fp, self.tn, self.fn))

        acc = ((tp + tn) / (tp + tn + fp + fn + self.eps)).cpu().numpy()
        prec = (tp / (tp + fp + self.eps)).cpu().numpy()
        rec = (tp / (tp + fn + self.eps)).cpu().numpy()

        y_true = torch.cat(self._all_targets, dim=0).numpy()  # [N, k]
        y_score = torch.cat(self._all_scores, dim=0).numpy()  # [N, k]

        auc = np.full(self.k, np.nan, dtype=float)
        for j in range(self.k):
            yj = y_true[:, j]
            if yj.min() != yj.max():
                auc[j] = roc_auc_score(yj, y_score[:, j])

        TP, FP, TN, FN = tp.sum(), fp.sum(), tn.sum(), fn.sum()

        acc_total = ((TP + TN) / (TP + TN + FP + FN + self.eps)).item()
        prec_total = (TP / (TP + FP + self.eps)).item()
        rec_total = (TP / (TP + FN + self.eps)).item()

        y_true_flat = y_true.ravel()
        y_score_flat = y_score.ravel()
        if y_true_flat.min() != y_true_flat.max():
            auc_total = roc_auc_score(y_true_flat, y_score_flat)
        else:
            auc_total = float('nan')

        per_class_results = {
            "accuracy": acc, "precision": prec, "recall": rec, "auc": auc,
        }

        return acc_total, prec_total, rec_total, auc_total, per_class_results