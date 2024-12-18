############################### import

import sys
import json
from anndata import AnnData
import cpa
import scanpy as sc
import importlib
import numpy as np
from tqdm import tqdm
import pickle
import os
import shutil
import torch
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.append("/data1/lichen/code/single_cell_perturbation/scPerturb/Byte_Pert_Data/")
import v1
from v1.utils import *
from v1.dataloader import *

import argparse

# importlib.reload(v1)
# importlib.reload(v1.utils)
# importlib.reload(v1.dataloader)

############################### function

def cpa_single_pert(pert, model_mode='whole'):
    
    
    # - get adata_pert and adata_ctrl
    adata_pert = adata_K562_sub[adata_K562_sub.obs['perturbation_group']==pert+' | K562']
    adata_ctrl = adata_K562_sub[list(adata_pert.obs['control_barcode'])]

    adata_pert = adata_pert[:, common_var]
    adata_ctrl = adata_ctrl[:, common_var]


    # - get adata_rna_common
    adata_rna_common = adata_rna[:, common_var]

    # - generate adata_train to input to scGen model
    np_list, obs_list, pert_list, celltype_list = [], [], [], []
    pert_list_2 = []
    adata_list = [adata_pert, adata_rna_common, adata_ctrl, adata_rna_common]
    for j, adata_ in enumerate(adata_list):
        if j in [0, 1]:
            pert_list.extend(['stimulated']*len(adata_))
        else:
            pert_list.extend(['ctrl']*len(adata_))

        if j in [0, 2]:
            celltype_list.extend(['K562']*len(adata_))
        else:
            celltype_list.extend([cell_line_bulk]*len(adata_))
        obs_list.extend([obs+f'_{j}' for obs in adata_.obs_names])
        
        if not isinstance(adata_.X, np.ndarray):
            np_list.append(adata_.X.toarray())
        else:
            np_list.append(adata_.X)

    adata_train = AnnData(X = np.vstack(np_list))
    adata_train.obs_names = obs_list
    adata_train.var_names = adata_pert.var_names

    adata_train.obs['condition'] = pert_list
    # adata_train.obs['condition_2'] = pert_list_2
    adata_train.obs['cell_type'] = celltype_list

    # - transform the adata_train.X to count
    adata_train.obs['cov_cond'] = adata_train.obs['cell_type'] + '_' + adata_train.obs['condition']
    adata_train.X = np.exp(adata_train.X)-1
    
    if model_prefix == 'CPA_v1':
        # - add norm
        sc.pp.normalize_per_cell(adata_train, key_n_counts='n_counts_all')

    # - initial model
    cpa.CPA.setup_anndata(adata_train, 
                        perturbation_key='condition',
                        control_group='ctrl',
                        #   dosage_key='dose',
                        categorical_covariate_keys=['cell_type'],
                        is_count_data=True,
                        #   deg_uns_key='rank_genes_groups_cov',
                        deg_uns_cat_key='cov_cond',
                        max_comb_len=1,
                        )

    # - set the train and validation for cpa
    # -- get total obs_names of the pert
    adata_train_new = adata_train[~((adata_train.obs["cell_type"] == cell_line_bulk) &
                        (adata_train.obs["condition"] == "stimulated"))].copy()
    # obs_df_split = adata_train_new.obs
    obs_df_sub_idx = np.array(adata_train_new.obs.index)

    np.random.seed(2024)
    np.random.shuffle(obs_df_sub_idx)

    # -- data split
    split_point_1 = int(len(obs_df_sub_idx) * 0.9)
    split_point_2 = int(len(obs_df_sub_idx) * (0.9+0.1))
    train = obs_df_sub_idx[:split_point_1]
    valid = obs_df_sub_idx[split_point_1:split_point_2]


    adata_train.obs['split_key'] = 'ood'

    # -- set the test row
    adata_train.obs.loc[train,'split_key'] = 'train'
    adata_train.obs.loc[valid,'split_key'] = 'valid'

    # - initial the model and training   
    model = cpa.CPA(adata=adata_train, 
                    split_key='split_key',
                    train_split='train',
                    valid_split='valid',
                    test_split='ood',
                    **model_params,
                )

    if cell_line_bulk == 'PC3' and adata_train.shape[0] == 726:
        batch_size = 512
    else:
        batch_size = 500
    model.train(max_epochs=2000,
                use_gpu=True, 
                batch_size=batch_size,
                plan_kwargs=trainer_params,
                early_stopping_patience=5,
                check_val_every_n_epoch=5,
                # save_path='../../datasets/',
                progress_bar_refresh_rate = 0
            )

    # - predict result
    model.predict(adata_train, batch_size=2048)

    # - get the pred data
    cat = cell_line_bulk + '_' + 'stimulated'
    cat_adata = adata_train[adata_train.obs['cov_cond'] == cat].copy()
    x_pred = cat_adata.obsm['CPA_pred']
    x_pred = np.log1p(x_pred)
    
    if model_prefix == 'CPA_v2': # normalize output
        x_pred = x_pred / x_pred.mean(1).reshape(-1,1) * adata_rna_common.X.mean(1).reshape(-1, 1)

    if model_mode == 'subset':
        # - get pert_gene_rank_dict
        adata_ctrl = adata_rna_common.copy()
        adata_pert = adata_ctrl.copy()
        adata_pert.X = x_pred
        
    elif model_mode == 'whole':
        # - get pert_gene_rank_dict
        adata_ctrl = adata_rna.copy()
        adata_pert = adata_ctrl.copy()
        adata_pert.X = x_pred[:, common_idx]
    else:
        raise ValueError()
        

    adata_pert.obs_names = [i+f'_{pert}' for i in adata_pert.obs_names]
    adata_ctrl.obs['batch'] = 'ctrl'
    adata_pert.obs['batch'] = 'pert'

    import anndata as ad
    adata_concat = ad.concat([adata_ctrl, adata_pert])

    # - cal de genes
    rankby_abs = False

    sc.tl.rank_genes_groups(
        adata_concat,
        groupby='batch',
        reference='ctrl',
        rankby_abs=rankby_abs,
        n_genes=len(adata_concat.var),
        use_raw=False,
        method = 'wilcoxon'
    )
    de_genes = pd.DataFrame(adata_concat.uns['rank_genes_groups']['names'])
    pvals = pd.DataFrame(adata_concat.uns['rank_genes_groups']['pvals'])
    pvals_adj = pd.DataFrame(adata_concat.uns['rank_genes_groups']['pvals_adj'])
    scores = pd.DataFrame(adata_concat.uns['rank_genes_groups']['scores'])
    logfoldchanges = pd.DataFrame(adata_concat.uns['rank_genes_groups']['logfoldchanges'])

    # - get gene_score
    gene_score = pd.DataFrame({'gene':list(de_genes['pert']),
                                'z-score':list(scores['pert'])})

    return (list(de_genes['pert']), list(scores['pert']))

############################### init
# - model initial
model_params = {
    "n_latent": 64,
    "recon_loss": "nb",
    "doser_type": "linear",
    "n_hidden_encoder": 128,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": True,
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": 6977,
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 0,
    "mixup_alpha": 0.0,
    "adv_steps": None,
    "n_hidden_adv": 64,
    "n_layers_adv": 3,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 5.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,
    "step_size_lr": 10,
}

# - get cell line name
common_cell_line = \
{   'A549': 'A549',
    'HEPG2': 'HepG2',
    'HT29': 'HT29',
    'MCF7': 'MCF7',
    # 'SKBR3': 'SK-BR-3',
    'SW480': 'SW480',
    'PC3': 'PC3',
    'A375': 'A375',
} # L1000 cell line : single-cell cell line

# - read adata_L1000, this is processed data
adata_L1000 = sc.read('/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark/benchmark_data/L1000/GSE92742/adata_gene_pert.h5ad')
adata_L1000

model_prefix = 'CPA_v2'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CPA on L1000")
    parser.add_argument('--cell_line_bulk', type=str, default=None)
    parser.add_argument('--model_mode', type=str, default='whole') # pretrain, init
    args = parser.parse_args()
    
    torch.cuda.set_device(1)

    print('reading adata_K562_sub')
    # - read adata_K562_sub
    adata_K562_sub = sc.read('/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark/benchmark_data/L1000/utils_data/adata_K562_sub.h5ad')

    # - get paras
    cell_line_bulk = args.cell_line_bulk
    cell_line_single = common_cell_line[cell_line_bulk]

    model_mode = args.model_mode


    print('='*20, f'cell line is {cell_line_single}')

    #===================prepare data
    if cell_line_bulk in ['PC3', 'A375']:
        save_dir_adata = '/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark/benchmark_data/L1000/single_cell_data/SCP542'
        adata_rna = sc.read(os.path.join(save_dir_adata, cell_line_bulk, f'adata_{cell_line_bulk}.h5ad'))
        
        # - read adata_rna_raw
        save_dir = f'/nfs/public/lichen/data/single_cell/cell_line/SCP542/process/{cell_line_bulk}'
        adata_rna_raw = sc.read(os.path.join(save_dir, f'adata.h5ad'))

    else:
        save_dir_adata = '/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark/benchmark_data/L1000/single_cell_data/CNP0003658'
        adata_rna = sc.read(os.path.join(save_dir_adata, cell_line_bulk, f'adata_{cell_line_bulk}.h5ad'))

        # - read adata_rna
        save_dir = f'/nfs/public/lichen/data/single_cell/cell_line/CNP0003658/process/RNA/{cell_line_single}'
        adata_rna_raw = sc.read(os.path.join(save_dir, f'adata_rna_{cell_line_single}.h5ad'))

    if not isinstance(adata_rna.X, np.ndarray):
        adata_rna.X = adata_rna.X.toarray()
        
    # - get common perts
    import json
    with open('/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark/benchmark_data/L1000/utils_data/direct_change_dict.json', 'r') as f:
        direct_change_dict = json.load(f)
    # - single_total_perts: the perts used in K562
    gene_list = direct_change_dict['gene_list']
    single_total_perts = list(direct_change_dict.keys())
    single_total_perts.remove('gene_list')

    # - get common pert
    adata_L1000_sub = adata_L1000[adata_L1000.obs['cell_id']==cell_line_bulk]
    L1000_total_perts = np.unique(adata_L1000_sub.obs['pert_iname'])
    common_perts = np.intersect1d(single_total_perts, L1000_total_perts)

    # - get common var
    common_var = np.intersect1d(adata_rna.var_names, direct_change_dict['gene_list'])
    common_var_2 = np.intersect1d(common_var, adata_L1000.var_names)

    print('common_perts num: ', len(common_perts))
    print('common var of direct change and single-cell data is: ', len(common_var))
    print('common var to L1000 data is: ', len(common_var_2))

    # 最近基因计算
    matrix = adata_rna.X.T
    index_list = np.array([list(adata_rna.var_names).index(i) for i in common_var])

    distance_matrix = cdist(matrix, matrix, metric='cosine')
    np.fill_diagonal(distance_matrix, np.inf)
    mask = np.ones(distance_matrix.shape, dtype=bool)
    mask[:, index_list] = False
    distance_matrix[mask] = np.inf
    nearest_indices = np.argmin(distance_matrix, axis=1)
    nearest_indices_list = nearest_indices.tolist()

    common_idx = [list(common_var).index(gene) if i in common_var else list(common_var).index(adata_rna.var_names[nearest_indices_list[i]]) for i, gene in enumerate(adata_rna.var_names)]

    torch.cuda.set_device(1)
    # - run CPA
    adata_pert_list = []
    pert_gene_rank_dict = {}
    for pert in tqdm(common_perts, desc='==========cpa_single_pert'):
        pert_gene_rank_dict[pert] = cpa_single_pert(pert, model_mode)
        # break
        
        
    save_dir = '/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark_202410/zero_shot/result'
    save_prefix = f'CPA_v1/{cell_line_bulk}' # use result of K562 to do the direct transfer
    # CPA: not norm for adata_train
    # CPA_v1: norm for adata_train
    # CPA_v2: not norm for adata_train, add norm for output
    os.makedirs(os.path.join(save_dir, save_prefix), exist_ok=True)

    import json
    with open(os.path.join(save_dir, save_prefix, 'pert_gene_rank_dict.json'), 'w') as f:
        json.dump(pert_gene_rank_dict, f)