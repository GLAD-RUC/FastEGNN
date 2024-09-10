#!/bin/bash

# 第一条命令
python -u generate_dataset_train.py --path '../3' --num-train 5000 --seed 43 --n_isolated 100 --n_stick 0 --n_hinge 0 --n_workers 32 --gaussians 3
while [ $? -ne 0 ]; do
    echo "第一条命令执行失败，正在重试..."
    python -u generate_dataset_train.py --path '../3' --num-train 5000 --seed 43 --n_isolated 100 --n_stick 0 --n_hinge 0 --n_workers 32 --gaussians 3
done

# 第二条命令
python -u generate_dataset_valid.py --path '../3' --num-train 2000 --seed 43 --n_isolated 100 --n_stick 0 --n_hinge 0 --n_workers 32 --gaussians 3
while [ $? -ne 0 ]; do
    echo "第二条命令执行失败，正在重试..."
    python -u generate_dataset_valid.py --path '../3' --num-train 2000 --seed 43 --n_isolated 100 --n_stick 0 --n_hinge 0 --n_workers 32 --gaussians 3
done

# 第三条命令
python -u generate_dataset_test.py --path '../3' --num-train 2000 --seed 43 --n_isolated 100 --n_stick 0 --n_hinge 0 --n_workers 32 --gaussians 3
while [ $? -ne 0 ]; do
    echo "第三条命令执行失败，正在重试..."
    python -u generate_dataset_test.py --path '../3' --num-train 2000 --seed 43 --n_isolated 100 --n_stick 0 --n_hinge 0 --n_workers 32 --gaussians 3
done
