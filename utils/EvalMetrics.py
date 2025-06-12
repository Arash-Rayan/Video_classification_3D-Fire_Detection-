from configs.config import args
from torch import nn
import torch
from torchmetrics import Accuracy , Recall


loss_fn = nn.BCEWithLogitsLoss()


def measure_model_performance(ypred , ytrue): 
    acc_fn = Accuracy(task='binary').to(args.device)
    acc = acc_fn(ypred , ytrue)

    sig_ypred = (torch.sigmoid(ypred) > 0.5).int()
    recallfn = Recall(task='binary').to(args.device)
    recall = recallfn(sig_ypred , ytrue)

    return acc , recall