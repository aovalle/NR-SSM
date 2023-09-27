import numpy as np
#from sklearn.metrics import f1_score as compute_f1_score

def calculate_accuracy(preds, y):
    preds = preds >= 0.5
    labels = y >= 0.5
    acc = preds.eq(labels).sum().float() / labels.numel()
    return acc


# def calculate_multiclass_f1_score(preds, labels):
#     f1score = compute_f1_score(labels, preds, average="weighted")
#     return f1score


def calculate_multiclass_accuracy(preds, labels):
    acc = float(np.sum((preds == labels).astype(int)) / len(labels))
    return acc