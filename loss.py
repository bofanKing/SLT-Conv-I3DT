# coding: utf-8
"""
Module to implement training loss
"""

import torch
from torch import nn, Tensor
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

class SignCLLoss(nn.Module):
    def __init__(self, temperature=0.07, max_negatives=10):
        super().__init__()
        self.temperature = temperature
        self.max_negatives = max_negatives

    def forward(self, features):  # [B, T, D]
        B, T, D = features.size()
        if T < 3:
            return torch.tensor(0.0, device=features.device)

        anchors = features[:, 1:T - 1, :]  # [B, T-2, D]
        positives = features[:, 2:T, :]  # [B, T-2, D]

        # Compute cosine similarity for anchor vs positive
        pos_sim = F.cosine_similarity(anchors, positives, dim=-1)  # [B, T-2]
        pos_sim = pos_sim / self.temperature  # [B, T-2]

        # Generate negatives by shifting along batch (for simplicity)
        with torch.no_grad():
            negatives = []
            for b in range(B):
                # Collect negative samples from other batches (excluding self)
                neg_pool = []
                for b_neg in range(B):
                    if b_neg != b:
                        neg_pool.append(features[b_neg])  # [T, D]
                neg_pool = torch.cat(neg_pool, dim=0)  # [(B-1)*T, D]
                if neg_pool.size(0) > self.max_negatives:
                    idx = torch.randperm(neg_pool.size(0))[:self.max_negatives]
                    neg_pool = neg_pool[idx]
                negatives.append(neg_pool)  # 每个 batch 得到 [K, D]

        # 合并所有 negatives 成 batch 维度的列表
        losses = []
        for b in range(B):
            anchor_batch = anchors[b]  # [T-2, D]
            pos_batch = pos_sim[b]  # [T-2]
            neg_batch = negatives[b]  # [K, D]
            if neg_batch.size(0) == 0:
                continue
            # [T-2, K] similarity matrix
            sim_neg = F.cosine_similarity(
                anchor_batch.unsqueeze(1),  # [T-2, 1, D]
                neg_batch.unsqueeze(0),  # [1, K, D]
                dim=-1
            ) / self.temperature
            exp_pos = torch.exp(pos_batch)  # [T-2]
            exp_neg = torch.exp(sim_neg).sum(dim=1)  # [T-2]
            loss = -torch.log(exp_pos / (exp_pos + exp_neg))  # [T-2]
            losses.append(loss.mean())

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=features.device)


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: distributions with batch*seq_len x vocab_size
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == targets.shape
            )
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )
        return loss
