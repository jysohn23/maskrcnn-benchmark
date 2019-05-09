# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            device = matched_idxs_per_image.device
            matched_idxs_per_image = matched_idxs_per_image.cpu()
            assert matched_idxs_per_image.dim() == 1
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min((matched_idxs_per_image > 0).sum(), num_pos)
            # TODO: investigate why torch.randperm with XLA device leads to loss nan.
            index_shuffle = torch.randperm(matched_idxs_per_image.size(0))
            matched_idxs_per_image_shuffle = matched_idxs_per_image[index_shuffle]
            labels_signed = torch.where(matched_idxs_per_image_shuffle <= 0,
                -matched_idxs_per_image_shuffle - 1, matched_idxs_per_image_shuffle)
            _, labels_shuffle_indices = torch.sort(labels_signed, descending=True)
            indices = index_shuffle[labels_shuffle_indices]
            pos_idx_per_image = indices[:num_pos]
            neg_idx_per_image = indices[-(self.batch_size_per_image - num_pos):]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            # protect against not enough positive and negative examples
            pos_idx_per_image_mask = pos_idx_per_image_mask & (matched_idxs_per_image > 0)
            neg_idx_per_image_mask = neg_idx_per_image_mask & (matched_idxs_per_image == 0)

            pos_idx.append(pos_idx_per_image_mask.to(device=device))
            neg_idx.append(neg_idx_per_image_mask.to(device=device))

        return pos_idx, neg_idx
