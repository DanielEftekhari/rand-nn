import torch

import metrics
import utils


def cross_entropy_loss_naive(logits, y):
    return -torch.sum(y * torch.log(utils.get_class_probs(logits)), dim=-1)


def cross_entropy_loss(logits, y):
    c = torch.max(logits, dim=-1)[0]
    z = c + torch.log(torch.sum(torch.exp(logits - c[:, None]), dim=-1))
    return -torch.sum(y * (logits - z[:, None]), dim=-1)


def kl_p_to_u(logits, y=None):
    max_entropy = metrics.max_entropy(logits.shape[1])
    return max_entropy - metrics.entropy(utils.get_class_probs(logits))


def kl_u_to_p(logits, y=None):
    max_entropy = metrics.max_entropy(logits.shape[1])
    return -max_entropy - 1. / logits.shape[1] * torch.sum(torch.log(utils.get_class_probs(logits)), dim=-1)


def kl_y_to_p(logits, y):
    y_logy = torch.sum(utils.where(y > 0, y * torch.log(y)), dim=-1)
    y_logp = torch.sum(utils.where(y > 0, y * torch.log(utils.get_class_probs(logits))), dim=-1)
    return y_logy - y_logp
