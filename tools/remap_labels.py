# -*- coding: utf-8 -*-
"""
tools/remap_labels.py
将 .label 文件从 SemanticKITTI raw ID 重映射为自定义类别 ID。

用法：
  # 原地重映射（覆盖原文件）
  python tools/remap_labels.py /root/autodl-tmp/260608_raw_16lidar/sequences/00/predictions

  # 输出到新目录（保留原文件）
  python tools/remap_labels.py /path/to/predictions -o /path/to/labels

  # 指定自定义映射文件
  python tools/remap_labels.py /path/to/predictions -m my_mapping.yaml

映射文件格式（YAML）：
  sk_raw_id: custom_id
  0: 0     # unlabeled → 未标注
  10: 1    # car → 汽车
  40: 4    # road → 道路
  50: 5    # building → 建筑
  51: 8    # fence → 低矮障碍物
  53: 7    # curb → 路沿
  70: 6    # vegetation → 植被
  71: 6    # trunk → 植被
  72: 8    # terrain → 低矮障碍物
  80: 8    # pole → 低矮障碍物
  99: 8    # other-object → 低矮障碍物
"""

import argparse
import os
import sys

import numpy as np
import yaml


DEFAULT_MAPPING = {
    0: 0,
    10: 1,
    40: 4,
    50: 5,
    51: 8,
    53: 7,
    70: 6,
    71: 6,
    72: 8,
    80: 8,
    99: 8,
}


def build_lut(mapping: dict) -> np.ndarray:
    lut = np.zeros(65536, dtype=np.uint32)
    for raw_id, custom_id in mapping.items():
        lut[int(raw_id)] = int(custom_id)
    return lut


def load_mapping(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        if 'mapping' in data:
            return {int(k): int(v) for k, v in data['mapping'].items()}
        return {int(k): int(v) for k, v in data.items()}
    raise ValueError(f'无法解析映射文件: {path}')


def remap_dir(input_dir: str, output_dir: str, lut: np.ndarray, dry_run: bool = False):
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith('.label') or f.endswith('.bin')
    )
    if not files:
        label_files = [f for f in os.listdir(input_dir) if f.endswith('.label')]
        if not label_files:
            print(f'[WARN] 目录中无 .label 文件: {input_dir}')
            return
        files = sorted(label_files)

    os.makedirs(output_dir, exist_ok=True)
    for fname in files:
        in_path = os.path.join(input_dir, fname)
        data = np.fromfile(in_path, dtype=np.uint32)
        remapped = lut[data]

        if dry_run:
            unique_before = np.unique(data)
            unique_after = np.unique(remapped)
            print(f'  {fname}: {list(unique_before)} -> {list(unique_after)}')
            continue

        out_path = os.path.join(output_dir, fname)
        remapped.tofile(out_path)

    if not dry_run:
        print(f'完成: {len(files)} 个文件  {input_dir} -> {output_dir}')


def main():
    p = argparse.ArgumentParser(
        description='将 .label 文件从 SK raw ID 重映射为自定义类别 ID')
    p.add_argument('input_dir', help='输入目录（含 .label 文件）')
    p.add_argument('-o', '--output_dir', default=None,
                   help='输出目录（默认原地覆盖）')
    p.add_argument('-m', '--mapping', default=None,
                   help='YAML 映射文件路径（不指定则使用默认 10→9 类映射）')
    p.add_argument('--dry-run', action='store_true',
                   help='预览映射变化，不实际写入')
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f'[ERROR] 目录不存在: {args.input_dir}')
        sys.exit(1)

    if args.mapping:
        mapping = load_mapping(args.mapping)
    else:
        mapping = DEFAULT_MAPPING

    lut = build_lut(mapping)
    output_dir = args.output_dir if args.output_dir else args.input_dir

    print(f'映射规则 ({len(mapping)} 条):')
    for raw_id, custom_id in sorted(mapping.items()):
        print(f'  SK raw {raw_id:>4d} -> custom {custom_id}')

    if args.dry_run:
        print('\n[预览模式 — 不写入文件]\n')

    remap_dir(args.input_dir, output_dir, lut, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
