# coding=utf-8
# Copyright (c) DLUP Contributors
import torch
import torch.nn.functional as F
from torch import nn


class VarAttention(nn.Module):
    def __init__(self, ndim=1024, n_classes = 2, **kwargs):
        super(VarAttention, self).__init__()
        
        self.num_classes = n_classes
        size = [ndim, 512, 1]
        self.L = size[0]
        self.D = size[1]
        self.K = size[2]
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.L * self.K, self.num_classes) # 2x since we also have variance
        )
        
        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))

    def compute_weighted_std(self, A, H, M):
        #TODO Now implemented to work with output as given above which is only for batch size of 1
        A, H = A.unsqueeze(2), H.unsqueeze(0)
        # Following https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
        # A: Attention (weight):    batch x instances x 1
        # H: Hidden:                batch x instances x channels
        H = H.permute(0, 2, 1)  # batch x channels x instances

        # M: Weighted average:      batch x channels
        M = M.unsqueeze(dim=2)  # batch x channels x 1
        # ---> S: weighted stdev:   batch x channels

        # N is non-zero weights for each bag: batch x 1

        N = (A != 0).sum(dim=1)
        
        upper = torch.einsum('abc, adb -> ad', A, (H - M) ** 2)  # batch x channels
        lower = ((N - 1) * torch.sum(A, dim=1)) / N  # batch x 1

        # Square root leads to infinite gradients when input is 0
        # Solution: No square root, or add eps=1e-8 to the input
        # But adding the eps will still lead to a large gradient from the sqrt through these 0-values.
        # Whether we look at stdev or variance shouldn't matter much, so we choose to go for the variance.
        S = (upper / lower)

        return S
    
    def forward(self, x):
        # Since we have batch_size = 1, squeezes from (1,num,features) to (num, features)
        H = x.squeeze(0)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        S = self.compute_weighted_std(A, H, M)

        MS = torch.cat((M, S), dim=1)  # concatenate the two tensors among the feature dimension, giving a twice as big feature

        logits  = self.classifier(MS)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict
