import torch

import utils


def cross_entropy_loss_naive(logits, y):
    losses = -torch.sum(y * torch.log(utils.get_class_probs(logits)), dim=-1)
    return losses


def cross_entropy_loss(logits, y):
    c = torch.max(logits, dim=-1)[0]
    z = c + torch.log(torch.sum(torch.exp(logits - c[:, None]), dim=-1))
    losses = -torch.sum(y * (logits - z[:, None]), dim=-1)
    return losses
