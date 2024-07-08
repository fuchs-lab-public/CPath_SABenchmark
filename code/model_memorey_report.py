import os
import glob
import torch
import time
from torch.cuda import amp
import argparse
import numpy as np
# import gc

import csv
import time
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from aggregator import GMA, TransMIL, DSMIL, CLAM_SB, CLAM_MB, VarAttention, GTP, PatchGCN_Surv, DeepGraphConv_Surv, MIL_Sum_FC_surv, MIL_Attention_FC_surv, MIL_Cluster_FC_surv, PureTransformer

parser = argparse.ArgumentParser()

#I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='output directory')
parser.add_argument('--method', type=str, default='', choices=[
    'GMA',
    'CLAM_SB',
    'CLAM_MB',
    'transMIL',
    'DSMIL',
    'VarMIL',
    'GTP',
    'PatchGCN',
    'DeepGraphConv',
    'MIL_Cluster_FC',
    'MIL_Attention_FC',
    'MIL_Sum_FC',
    'ViT_MIL',
    'DTMIL',
], help='which aggregation method to use')
parser.add_argument('--data', type=str, default='', choices=[
    'msk_lung_egfr',
    'msk_lung_io',
    'sinai_breast_cancer',
    'sinai_breast_er',
    'sinai_lung_egfr',
    'sinai_ibd_detection',
    'biome_breast_hrd',
    'llovet_hcc_io',
], help='which data to use')
parser.add_argument('--encoder', type=str, default='', choices=[
    'tres50_imagenet',
    'dinosmall',
], help='which encoder to use')

parser.add_argument('--ndim', default=512, type=int, help='output dimension of feature extractor')

def get_aggregator(method='', ndim=1024, **kwargs):
    # GMA
    if method == 'GMA':
        return GMA(ndim=ndim, **kwargs)
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
    elif method == 'DSMIL':
        return DSMIL(ndim=ndim, n_classes=2, **kwargs)
    # DeepSMILE
    elif method == 'VarMIL':
        return VarAttention(ndim=ndim, **kwargs)
    # GTP
    elif method == 'GTP':
        return GTP(ndim=ndim, **kwargs)
    # PatchGCN
    elif method == 'PatchGCN':
        return PatchGCN_Surv(ndim=ndim, n_classes=2, **kwargs)
    # DeepGraphConv
    elif method == 'DeepGraphConv':
        return DeepGraphConv_Surv(ndim=ndim, n_classes=2, **kwargs)
    elif method == 'MIL_Sum_FC':
        return MIL_Sum_FC_surv(ndim=ndim, n_classes=2, **kwargs)
    elif method == 'MIL_Attention_FC':
        return MIL_Attention_FC_surv(ndim=ndim, n_classes=2, **kwargs)
    elif method == 'MIL_Sum_FC':
        return MIL_Sum_FC_surv(ndim=ndim, n_classes=2, **kwargs)
    elif method == 'MIL_Cluster_FC':
        return MIL_Cluster_FC_surv(ndim=ndim, n_classes=2, **kwargs)
    # elif method == 'DTMIL':
    #     return DTMIL(ndim=ndim, n_classes=2, **kwargs)
    else:
        raise Exception(f'Method {method} not defined')
    
def find_max_min_file(directory):
    """Returns the paths of the smallest and largest .pth files in the given directory."""
    pth_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if os.path.isfile(os.path.join(directory, f)) and f.endswith('.pth')]
    min_file = min(pth_files, key=os.path.getsize)
    max_file = max(pth_files, key=os.path.getsize)
    return min_file, max_file


def run(model, data, device, use_amp=True):
    # Move the model and data to the appropriate device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() and use_amp else 'cpu')
    model.to(device)
    data = data.squeeze(0).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = amp.GradScaler(enabled=use_amp)

    start_time = time.time()

    # Model training logic (for one epoch as an example)
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the parameter gradients

    criterion = nn.CrossEntropyLoss().to(device)
    target = torch.randint(0, 2, (data.size(0),), device=device).long()

    # Forward pass with Automatic Mixed Precision
    # with amp.autocast(enabled=use_amp):
    outputs = model(data)
    logits = outputs['logits']
    loss = criterion(logits, target)

    print("loss")

    # Backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # End the timer
    end_time = time.time()
    runtime = end_time - start_time

    # Reset peak memory stats after each run
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Reset peak memory stats after each run
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    # Get model summary for the number of parameters (which correlates with model size)
    model_summary = summary(model, input_size=(data.shape[0], data.shape[1], data.shape[2]), verbose=0)
    model_size = sum(p.numel() for p in model.parameters())
    
    # Record maximum memory allocated for CPU and GPU
    if device.type == 'cuda':
        memory_gpu = torch.cuda.max_memory_allocated(device)
    else:
        memory_gpu = None
    memory_cpu = sum(p.numel() for p in model.parameters() if not p.is_cuda) * data.element_size()

    torch.cuda.reset_peak_memory_stats(device)  # Reset memory stats for next run

    # Return the collected metrics
    return {
        'runtime': runtime,
        'model_size': model_size,
        'memory_gpu': memory_gpu,
        'memory_cpu': memory_cpu,
        'model_summary': str(model_summary)
    }

def suggest_memory_requirements(memory_usage_data):
    # Define thresholds for memory usage for different GPUs
    thresholds = {
        'v100': 16 * 1024**3,  # 16GB in bytes
        'a100': 40 * 1024**3,  # 40GB in bytes
        'h10080g': 80 * 1024**3  # 80GB in bytes
    }
    
    # Suggestion based on the maximum GPU memory recorded
    max_memory_gpu = max(memory_usage_data['memory_gpu'])
    suggested_gpu = min((gpu for gpu, mem in thresholds.items() if mem >= max_memory_gpu), key=lambda x: thresholds[x], default='h10080g')
    
    # Return the suggested GPU type
    return suggested_gpu

# Assuming save_memory_suggestions is a function you already have
def save_memory_suggestions(suggestion):
    # Save the suggestion to a text file or handle it as needed
    with open('memory_suggestions.txt', 'w') as f:
        f.write(suggestion)

def run_benchmarking(data, encoder, ndim, method_list):
    default_path = '/sc/arion/projects/comppath_500k/SAbenchmarks/data/'
    directory_path = os.path.join(default_path, data, 'tensors', encoder)
    
    print(f"start finding largest files at {directory_path}")
    min_file, max_file = find_max_min_file(directory_path)
    min_tile_no = torch.load(min_file)['features'].shape[0]
    max_tile_no = torch.load(max_file)['features'].shape[0]
    print(f"Minimum tile number: {min_tile_no}, Maximum tile number: {max_tile_no}")

    print(f"torch cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    min_tile_no = 30
    max_tile_no = 500
    tile_numbers = np.linspace(min_tile_no, max_tile_no, 5, dtype=int)  

    csv_save_path = os.path.join(directory_path, 'model_report.csv')
    with open(csv_save_path, 'w', newline='') as file:
        print(f"start writing csv file at {csv_save_path}")
        writer = csv.DictWriter(file, fieldnames=['data', 'encoder', 'method', 'amp', 'memory_gpu', 'suggested_gpu'])
        writer.writeheader()

        for method in method_list:
            print(f"Testing {method}")
            aggregator = get_aggregator(method=method, ndim=ndim)
            
            for amp in [False, True]:
                for tile_no in tile_numbers:
                    # Generate pseudo_tensor for the current tile number
                    pseudo_tensor = torch.randn(1, tile_no, ndim)
                    
                    metrics = run(aggregator, pseudo_tensor, device, use_amp=amp)
                    metrics.update({
                        'tile_no':tile_no,
                        'data': data,
                        'encoder': encoder,
                        'method': method,
                        'amp': amp,
                        # Uncomment or modify the 'suggested_gpu' line as needed
                        # 'suggested_gpu': suggest_memory_requirements(metrics['memory_gpu'])
                    })
                    writer.writerow(metrics)
                    
                    # # Explicitly delete large variables and clear PyTorch cache
                    # del pseudo_tensor
                    # gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

if __name__ == '__main__':
    print("start working")
    args = parser.parse_args()
    
    # Dim of features
    if args.encoder.startswith('tres50'):
        args.ndim = 1024
    elif args.encoder.startswith('dinosmall'):
        args.ndim = 384
    elif args.encoder.startswith('dinobase'):
        args.ndim = 768
    
    method_list = ['GMA']
    # method_list = ['GMA','CLAM_SB','CLAM_MB','transMIL','DSMIL','VarMIL','GTP','PatchGCN','DeepGraphConv','MIL_Attention_FC','MIL_Sum_FC','ViT_MIL']
    run_benchmarking(args.data, args.encoder, args.ndim, method_list)
    # run_benchmarking(data, encoder, ndim, method_list)

    