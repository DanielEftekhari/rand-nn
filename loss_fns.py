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


def kl_p_to_u(logits, y):
    max_ent = utils.max_ent(y.shape[1])
    return max_ent - utils.entropy(logits)


def kl_u_to_p(logits, y):
    max_ent = utils.max_ent(y.shape[1])
    return -max_ent - 1. / y.shape[1] * torch.sum(utils.get_class_probs(logits), dim=-1)
