# =====================================================================================
# This Python script is modified based on the work available at https://github.com/vkola-lab/tmi2022/blob/main/models/GraphTransformer.py.
# Original repository: https://github.com/vkola-lab/tmi2022/tree/main
# 
# The aggregation model used in this project is adapted from:
# "A Graph-Transformer for Whole Slide Image Classification"
# by Yi Zheng, Rushin H. Gindra, Emily J. Green, Eric J. Burks, Margrit Betke, Jennifer E. Beane, Vijaya B. Kolachalama
# Citation: Zheng, Y., Gindra, R.H., Green, E.J., Burks, E.J., Betke, M., Beane, J.E., Kolachalama, V.B. (2022).
# IEEE Transactions on Medical Imaging, 41(11), 3003-3015, DOI: 10.1109/TMI.2022.3176598
# =====================================================================================

import sys
import os
import torch
import random
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .gtp_utils.ViT import *
from .gtp_utils.gcn import GCNBlock

from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch.nn import Linear

class GTP(nn.Module):
    def __init__(self, ndim=1024, n_class = 2):
        super(GTP, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100
        
        self.transformer = VisionTransformer(num_classes=n_class, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = nn.CrossEntropyLoss()
        
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(ndim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding,0.,0)       # 64->128
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)                                          # 100-> 20

    def forward(self, h, adj, mask, is_print=False, graphcam_flag=False):
        node_feat = h
        
        cls_loss=node_feat.new_zeros(self.num_layers)
        rank_loss=node_feat.new_zeros(self.num_layers-1)
        X = node_feat
        p_t=[]
        pred_logits=0
        visualize_tools=[]
        # visualize_tools1=[labels.cpu()]
        embeds=0
        concats=[]
        
        layer_acc=[]
        
        X = mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        s = self.pool1(X)
                
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)
        
        logits = self.transformer(X)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'mc1': mc1, 'o1': o1}
        return results_dict