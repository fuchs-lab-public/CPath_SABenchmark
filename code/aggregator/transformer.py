# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

class PureTransformer(nn.Module):
    def __init__(self, ndim=1024, n_classes=2):
        super(PureTransformer, self).__init__()

        self.transformer = nn.Transformer(
            d_model=128,
            num_encoder_layers=4,
            num_decoder_layers=2,
            dim_feedforward=256,
        )

        self.linear_proj = nn.Linear(ndim, 128)

        self.cls_token = nn.Parameter(torch.rand(1, 128))

        self.mlp = nn.Linear(128, n_classes)

    def forward(self, tensors):
        # src: (S, N, E)(S,N,E)
        # tgt: (T, N, E)(T,N,E)
        # src_mask: (S, S)(S,S)
        x = tensors
        #mask = sampler.mask  # N * S
        b, h, w, c = x.shape
        x = x.view(b, c, -1)
        x = x.permute(0, 2, 1)  # N S E

        x = self.linear_proj(x)  # N E S

        tgt = self.cls_token
        tgt = tgt.unsqueeze(0).expand(b, -1, -1)

        # print(x.shape, tgt.shape)
        x = x.permute(1, 0, 2)  # S N E
        tgt = tgt.permute(1, 0, 2)  # S N E
        out = self.transformer(src=x, tgt=tgt)

        logit = self.mlp(out)

        logits = logit.squeeze(0)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict
