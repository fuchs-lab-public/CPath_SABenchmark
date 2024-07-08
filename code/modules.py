import os
import numpy as np
import torch
# Import aggregation methods
from aggregator import GMA, TransMIL, DSMIL, CLAM_SB, CLAM_MB, VarAttention, GTP, PatchGCN_Surv, DeepGraphConv_Surv, MIL_Sum_FC_surv, MIL_Attention_FC_surv, MIL_Cluster_FC_surv, PureTransformer
# PureTransformer, DTMIL

def get_aggregator(method='', ndim=1024, **kwargs):
    # GMA
    if method == 'AB-MIL':
        return GMA(ndim=ndim, **kwargs)
    elif method == 'AB-MIL_FC_small':
        return MIL_Attention_FC_surv(ndim=ndim, n_classes=2, size_arg='small', **kwargs)
    elif method == 'AB-MIL_FC_big':
        return MIL_Attention_FC_surv(ndim=ndim, n_classes=2, size_arg='big', **kwargs)
    # DeepSMILE
    elif method == 'VarMIL':
        return VarAttention(ndim=ndim, **kwargs)
    # CLAM
    if method == 'CLAM_SB':
        return CLAM_SB(ndim=ndim, **kwargs)
    elif method == 'CLAM_MB':
        return CLAM_MB(ndim=ndim, **kwargs)
    # TransMIL
    elif method == 'ViT_MIL':
        return PureTransformer(ndim=ndim, n_classes=2, **kwargs)
    elif method == 'transMIL':
        return TransMIL(ndim=ndim, **kwargs)
    # DSMIL
    elif method == 'DS-MIL':
        return DSMIL(ndim=ndim, n_classes=2, **kwargs)
    # GTP
    elif method == 'GTP':
        return GTP(ndim=ndim, **kwargs)
    # PatchGCN
    elif method == 'PatchGCN':
        return PatchGCN_Surv(ndim=ndim, n_classes=2, **kwargs)
    # DeepGraphConv
    elif method == 'DeepGraphConv':
        return DeepGraphConv_Surv(ndim=ndim, n_classes=2, **kwargs)
    # elif method == 'MIL_Sum_FC':
    #     return MIL_Sum_FC_surv(ndim=ndim, n_classes=2, **kwargs)
    # elif method == 'MIL_Cluster_FC':
    #     return MIL_Cluster_FC_surv(ndim=ndim, n_classes=2, **kwargs)
    # elif method == 'DTMIL':
    #     return DTMIL(ndim=ndim, n_classes=2, **kwargs)
    else:
        raise Exception(f'Method {method} not defined')
