# signcl_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SignCLLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features):  # features: [B, T, D]
        B, T, D = features.shape
        loss = 0.0
        count = 0

        for b in range(B):
            for t in range(1, T - 1):
                anchor = features[b, t]
                pos = features[b, t + 1]
                neg_pool = torch.cat([features[b, :t-1], features[b, t+2:]], dim=0)

                pos_sim = self.cosine_similarity(anchor, pos) / self.temperature
                if neg_pool.size(0) > 0:
                    neg_sim = self.cosine_similarity(anchor.unsqueeze(0), neg_pool) / self.temperature
                    neg_sim = torch.exp(neg_sim).sum()
                else:
                    continue  # skip if not enough negatives

                loss += -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim))
                count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=features.device)
