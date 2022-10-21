import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn import Module
import numpy as np


def train(model: Module, dataloader: DataLoader, optimizer, criterion, grad_clip: int = -1):
    losses = []
    model.train()
    y_preds = []
    y_trues = []
    for x, y, _ in dataloader:
        model.zero_grad()
        logits = model(x)
        logits = logits.reshape(y.shape)
        loss = criterion(logits, y.float())
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(loss.item())
        y_pred = torch.sigmoid(logits).round().int()
        for y_p, y_t in zip(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy()):
            y_preds.append(y_p)
            y_trues.append(y_t)

    return np.average(losses), y_preds, y_trues


def evaluate(model: Module, dataloader: DataLoader, criterion):
    losses = []
    model.eval()
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for x, y, _ in dataloader:
            logits = model(x)
            logits = logits.reshape(y.shape)
            loss = criterion(logits, y.float())

            losses.append(loss.item())
            y_pred = torch.sigmoid(logits).round().int()
            for y_p, y_t in zip(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy()):
                y_preds.append(y_p)
                y_trues.append(y_t)

    return np.average(losses), y_preds, y_trues
