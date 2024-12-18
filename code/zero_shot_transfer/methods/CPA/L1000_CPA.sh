#!/bin/bash

source /root/miniconda3/bin/activate scgpt_2
export LD_LIBRARY_PATH=/root/miniconda3/envs/scgpt_2/lib:$LD_LIBRARY_PATH

# 定义字符串列表
args_list=("A549" "HEPG2" "HT29" "MCF7" "SW480" "PC3" "A375")
model_mode="init"

# 遍历字符串列表，并行运行 Python 脚本
for arg in "${args_list[@]}"; do
    python /data1/lichen/code/single_cell_perturbation/scPerturb_202410/zero_shot/method/CPA/L1000_CPA_single_celltype.py --cell_line_bulk "$arg" &  # 使用 & 使其在后台运行
done

# 等待所有后台任务完成
wait