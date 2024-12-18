import sys
import json
from anndata import AnnData
# import method.CPA.cpa_func_lc as cpa_func_lc
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

def cpa_single_pert(pert, adata_ori):
    
    print(pert)
    
    global pert_data, adata_rna, common_var, cell_line_bulk, model_params, trainer_params
    # global pert_data
    
    print('============', adata_ori.shape)
    
    # - get adata_pert and adata_ctrl
    adata_pert = pert_data.adata_split[pert_data.adata_split[pert_data.adata_split.obs['perturbation_group']==pert+' | K562'].obs_names]
    adata_ctrl = pert_data.adata_split[adata_pert.obs['control_barcode']]

    adata_pert = adata_ori[adata_pert.obs_names, common_var]
    adata_ctrl = adata_ori[adata_ctrl.obs_names, common_var]


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
        np_list.append(adata_.X.toarray())

    adata_train = AnnData(X = np.vstack(np_list))
    adata_train.obs_names = obs_list
    adata_train.var_names = adata_pert.var_names

    adata_train.obs['condition'] = pert_list
    # adata_train.obs['condition_2'] = pert_list_2
    adata_train.obs['cell_type'] = celltype_list

    # - transform the adata_train.X to count
    adata_train.obs['cov_cond'] = adata_train.obs['cell_type'] + '_' + adata_train.obs['condition']
    adata_train.X = np.exp(adata_train.X)-1

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

    model.train(max_epochs=2000,
                use_gpu=True, 
                batch_size=500,
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


    # - get pert_gene_rank_dict
    adata_ctrl = adata_rna_common.copy()
    adata_pert = adata_ctrl.copy()
    adata_pert.X = x_pred
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




    
    


# 定义处理每个 cell_line_single 的函数
def process_cell_line(adata_ori, cell_line_bulk, cell_line_single, adata_L1000, gpu_id = 0):
    print('=' * 20, f'cell line is {cell_line_single}')
    
    torch.cuda.set_device(gpu_id)  # 设置每个进程使用不同的 GPU

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

    # - consctrut corr mtx
    if not isinstance(adata_rna.X, np.ndarray):
        adata_rna.X = adata_rna.X.toarray()
    corr_mtx = np.corrcoef(adata_rna.X.T)
    
    # - get var_names
    var_names = list(adata_rna.var_names)
    
    # - get common pert
    adata_L1000_sub = adata_L1000[adata_L1000.obs['cell_id']==cell_line_bulk]
    L1000_total_perts = np.unique(adata_L1000_sub.obs['pert_iname'])
    common_perts = np.intersect1d(adata_rna.var_names, L1000_total_perts)

    
    
    print('common_perts num: ', len(common_perts))
    print('common var to L1000 data is: ', len(np.intersect1d(var_names, adata_L1000.var_names)))

    
    adata_pert_list = []
    pert_gene_rank_dict = {}
    for pert in tqdm(common_perts, desc='cpa_single_pert'):
        pert_gene_rank_dict[pert] = cpa_single_pert(pert, adata_ori)
        
        
    save_dir = '/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark_202410/zero_shot/result'
    save_prefix = f'CPA/{cell_line_bulk}' # use result of K562 to do the direct transfer
    os.makedirs(os.path.join(save_dir, save_prefix), exist_ok=True)

    import json
    with open(os.path.join(save_dir, save_prefix, 'pert_gene_rank_dict.json'), 'w') as f:
        json.dump(pert_gene_rank_dict, f)


