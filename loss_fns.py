import torch

import metrics
import utils


def cross_entropy_loss_naive(logits, y):
    losses = -torch.sum(y * torch.log(utils.get_class_probs(logits)), dim=-1)
    return losses


def cross_entropy_loss(logits, y):
    c = torch.max(logits, dim=-1)[0]
    z = c + torch.log(torch.sum(torch.exp(logits - c[:, None]), dim=-1))
    losses = -torch.sum(y * (logits - z[:, None]), dim=-1)
    return losses


def kl_p_to_u(logits):
    max_ent = metrics.max_ent(logits.shape[1])
    return max_ent - metrics.entropy(utils.get_class_probs(logits))


def kl_u_to_p(logits):
    max_ent = metrics.max_ent(logits.shape[1])
    return -max_ent - 1. / logits.shape[1] * torch.sum(torch.log(utils.get_class_probs(logits)), dim=-1)


def kl_y_to_p(logits, y):
    y_logy = torch.sum(utils.where(y > 0, y * torch.log(y)), dim=-1)
    y_logp = torch.sum(utils.where(y > 0, y * torch.log(utils.get_class_probs(logits))), dim=-1)
    return y_logy - y_logp
