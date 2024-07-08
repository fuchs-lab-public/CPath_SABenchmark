import os
import pandas as pd
import torch
from pathlib import Path
    
def get_datasets(mccv=0, data='', encoder='', method=''):
    # Load slide data
    df = pd.read_csv(os.path.join('/sc/arion/projects/comppath_500k/SAbenchmarks/data', data, 'slide_data.csv'))
    df['tensor_path'] = [os.path.join(x.tensor_root, encoder, x.tensor_name) for _, x in df.iterrows()]
    spatial_root = Path(df.loc[0,"tensor_root"]).parent / "spatial"
    if method == 'GTP': # Graph with adjacency matrix
        df['method_tensor_path'] = [spatial_root / 'GTP' / x.tensor_name for _, x in df.iterrows()]
    elif method in ['PatchGCN', 'DeepGraphConv']: # Graph with edge_index
        df['method_tensor_path'] = [spatial_root / 'PatchGCN' / x.tensor_name for _, x in df.iterrows()]
    elif method in ['MIL_Cluster_FC']: # cluster
        df['method_tensor_path'] = [spatial_root / 'Cluster' / x.tensor_name for _, x in df.iterrows()]
    elif method in ['ViT_MIL','DTMIL']: # Positional Encoded Transformer
        df['method_tensor_path'] = [spatial_root / 'DT-MIL' / x.tensor_name for _, x in df.iterrows()]
    else:
        df['method_tensor_path'] = [os.path.join(x.tensor_root, encoder, x.tensor_name) for _, x in df.iterrows()]
    # Select mccv and clean
    df = df.rename(columns={'mccv{}'.format(mccv): 'mccvsplit'})[['slide', 'target', 'mccvsplit', 'tensor_path','method_tensor_path']]
    df['mccvsplit'] = df['mccvsplit'].fillna('test')
    # Split into train and val
    df_train = df[df.mccvsplit == 'train'].reset_index(drop=True).drop(columns=['mccvsplit'])
    df_val = df[df.mccvsplit == 'val'].reset_index(drop=True).drop(columns=['mccvsplit'])
    df_test = None
    if data in ['camelyon16']:
        df_test = df[df.mccvsplit == 'test'].reset_index(drop=True).drop(columns=['mccvsplit'])
    # Create my loader objects
    if method in ['GTP','PatchGCN', 'DeepGraphConv' ,'MIL_Cluster_FC','MIL_Sum_FC','ViT_MIL','DTMIL']:
        dset_train = slide_dataset_classification_graph(df_train)
        dset_val = slide_dataset_classification_graph(df_val)
        dset_test = slide_dataset_classification_graph(df_test) if df_test is not None else None
    else:
        dset_train = slide_dataset_classification(df_train)
        dset_val = slide_dataset_classification(df_val)
        dset_test = slide_dataset_classification(df_test) if df_test is not None else None
    
    return dset_train, dset_val, dset_test

class slide_dataset_classification(torch.utils.data.Dataset):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        data = torch.load(row.tensor_path)  # feature matrix and possibly other data
        try:
            feat = data['features']
        except:
            feat = data
        return {'features': feat, 'target': row.target}

class slide_dataset_classification_graph(slide_dataset_classification):
    def __init__(self, df):
        super(slide_dataset_classification_graph, self).__init__(df)

    def __getitem__(self, index):
        # Load data using the parent class method
        item = super(slide_dataset_classification_graph, self).__getitem__(index)
        # Additional graph-specific data extraction
        data = torch.load(self.df.iloc[index].method_tensor_path)
        if 'adj_mtx' in data:
            item['adj_mtx'] = data['adj_mtx']
            item['mask'] = data['mask']
        if 'edge_latent' in data.keys():
            item['edge_index'] = data['edge_index']
            item['edge_latent'] = data['edge_latent']
            item['centroid'] = data['centroid']
        if 'feat_map' in data:
            item['feat_map'] = data['feat_map']
            item['mask'] = data['mask']
        return item
