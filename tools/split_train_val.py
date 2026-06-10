"""
按序列内随机比例分割 train/val 帧清单。

用法：
    python tools/split_train_val.py /root/autodl-tmp/16lidar_labeled.v.tcn/ -s 00,01,02,03,04 --val-ratio 0.3 -o runs/finetune_tcn/

输出：
    runs/finetune_tcn/train_frames.txt   (每序列 70% 帧)
    runs/finetune_tcn/val_frames.txt     (每序列 30% 帧)
"""

import argparse
import os
import random
import sys


def main():
    p = argparse.ArgumentParser(
        description='按序列内随机比例分割 train/val 帧清单')
    p.add_argument('root', help='数据集根目录（含 sequences/）')
    p.add_argument('-s', '--seqs', default='00,01,02,03,04',
                   help='序列列表，逗号分隔（默认 00,01,02,03,04）')
    p.add_argument('--val-ratio', type=float, default=0.3,
                   help='验证集比例（默认 0.3）')
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子（默认 42）')
    p.add_argument('-o', '--out-dir', default='.',
                   help='输出目录（默认当前目录）')
    args = p.parse_args()

    random.seed(args.seed)
    seq_list = [s.strip() for s in args.seqs.split(',') if s.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, 'train_frames.txt')
    val_path   = os.path.join(args.out_dir, 'val_frames.txt')

    train_lines = []
    val_lines = []
    total = 0
    seq_stats = {}

    for seq in seq_list:
        velo_dir = os.path.join(args.root, 'sequences', seq, 'velodyne')
        if not os.path.isdir(velo_dir):
            print(f'[WARN] 序列 {seq} 不存在: {velo_dir}')
            continue

        bins = sorted(f for f in os.listdir(velo_dir) if f.endswith('.bin'))
        stems = [b.replace('.bin', '') for b in bins]
        random.shuffle(stems)

        n_val = max(1, int(len(stems) * args.val_ratio))
        val_stems = sorted(stems[:n_val])
        train_stems = sorted(stems[n_val:])

        for s in train_stems:
            train_lines.append(f'{seq}/{s}\n')
        for s in val_stems:
            val_lines.append(f'{seq}/{s}\n')

        total += len(stems)
        seq_stats[seq] = (len(train_stems), len(val_stems))
        print(f'  序列 {seq}: {len(stems)} 帧 → train={len(train_stems)}, val={len(val_stems)}')

    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print(f'\n总计 {total} 帧 → train={len(train_lines)}, val={len(val_lines)}')
    print(f'Train 清单: {train_path}')
    print(f'Val   清单: {val_path}')


if __name__ == '__main__':
    main()
