# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    loss = smooth_l1_loss_no_reduction(input, target, beta, size_average)
    if size_average:
        return loss.mean()
    return loss.sum()


def smooth_l1_loss_no_reduction(input, target, beta=1. / 9, size_average=True):
    """
    Very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    return torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
