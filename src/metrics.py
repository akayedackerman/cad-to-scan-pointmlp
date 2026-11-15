"""Classification metrics used by both the supervised and zero-shot evaluations."""
import numpy as np


def confusion(y, pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def summarize(y, pred, n_classes):
    """Overall accuracy (OA), mean class accuracy (mAcc) and per-class accuracy."""
    y, pred = np.asarray(y), np.asarray(pred)
    cm = confusion(y, pred, n_classes)
    per_class = (np.diag(cm) / np.maximum(cm.sum(1), 1)).tolist()
    present = cm.sum(1) > 0
    return {"OA": float((y == pred).mean()), "mAcc": float(np.mean(np.array(per_class)[present])),
            "n": int(len(y)), "per_class": per_class}
