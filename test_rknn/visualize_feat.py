#!/usr/bin/env python3
"""
可视化并对比 RKNN 和 ONNX 的 backbone 输出特征图。
将 rv_rknn.npy / rv_onnx.npy 从板端拷回 PC 后运行。

用法：
  # 板端：在 caf_node callback 里临时加两行保存一帧：
  #   np.save('/tmp/rv_rknn.npy', rv_feat)   # shape (1, 64, 16, 1024)
  #   np.save('/tmp/pb_rknn.npy', pb_feat)
  #   np.save('/tmp/rv_onnx.npy', rv_feat_ref)  # 如果有 ONNX 对比
  #
  # PC 端：
  #   python3 visualize_feat.py --rknn rv_rknn.npy [--onnx rv_onnx.npy]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_feat(feat, title, out_path, max_cols=8):
    """feat: (C, H, W)，按通道画 heatmap"""
    C, H, W = feat.shape
    rows = (C + max_cols - 1) // max_cols
    fig, axes = plt.subplots(rows, max_cols,
                              figsize=(max_cols * 2, rows * (H / W * 2 + 0.5)))
    fig.suptitle(title, fontsize=10)
    for i in range(rows * max_cols):
        ax = axes[i // max_cols][i % max_cols] if rows > 1 else axes[i % max_cols]
        ax.axis('off')
        if i >= C:
            continue
        ch = feat[i]
        vmax = np.percentile(np.abs(ch), 99)
        ax.imshow(ch, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto',
                  interpolation='nearest')
        ax.set_title(f'ch{i}\n{ch.mean():.2f}±{ch.std():.2f}', fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'保存: {out_path}')

def plot_channel_mean(feat, title, out_path):
    """feat: (C, H, W)，画 channel-mean image（综合所有通道的平均激活）"""
    mean_img = feat.mean(axis=0)   # (H, W)
    std_img  = feat.std(axis=0)    # (H, W)
    fig, axes = plt.subplots(1, 2, figsize=(16, 2))
    fig.suptitle(title, fontsize=9)
    im0 = axes[0].imshow(mean_img, cmap='viridis', aspect='auto')
    axes[0].set_title(f'channel mean  [{mean_img.min():.2f}, {mean_img.max():.2f}]')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(std_img, cmap='hot', aspect='auto')
    axes[1].set_title(f'channel std  [{std_img.min():.2f}, {std_img.max():.2f}]')
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'保存: {out_path}')

def plot_diff(rknn, onnx, out_path):
    """对比 RKNN 和 ONNX 的逐通道差异"""
    diff = np.abs(rknn - onnx)   # (C, H, W)
    C = diff.shape[0]
    ch_mean = diff.mean(axis=(1, 2))   # (C,)
    ch_max  = diff.max(axis=(1, 2))    # (C,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('RKNN vs ONNX 差异', fontsize=10)

    # 左：各通道平均误差柱状图
    colors = ['red' if v > 0.5 else 'steelblue' for v in ch_mean]
    axes[0].bar(range(C), ch_mean, color=colors)
    axes[0].axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='threshold=0.5')
    axes[0].set_xlabel('channel')
    axes[0].set_ylabel('mean |diff|')
    axes[0].set_title(f'各通道平均误差（红色>0.5，共{(ch_mean>0.5).sum()}个通道）')
    axes[0].legend()

    # 右：差异最大的通道的空间分布
    worst_ch = int(ch_mean.argmax())
    diff_img = diff[worst_ch]
    im = axes[1].imshow(diff_img, cmap='hot', aspect='auto')
    axes[1].set_title(f'最大误差通道 ch{worst_ch} 的空间分布\nmean={ch_mean[worst_ch]:.4f} max={ch_max[worst_ch]:.4f}')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'保存: {out_path}')

    # 打印各通道误差排名
    print('\n各通道误差排名（top 10）:')
    top = np.argsort(ch_mean)[::-1][:10]
    for i, c in enumerate(top):
        print(f'  #{i+1} ch{c:2d}: mean_diff={ch_mean[c]:.4f}  max_diff={ch_max[c]:.4f}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rknn', required=True, help='rv_rknn.npy 路径')
    ap.add_argument('--onnx', default=None,  help='rv_onnx.npy 路径（可选）')
    ap.add_argument('--out',  default='.',   help='输出目录')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rknn = np.load(args.rknn)
    if rknn.ndim == 4:
        rknn = rknn[0]   # (1, C, H, W) -> (C, H, W)
    print(f'RKNN feat: shape={rknn.shape}  min={rknn.min():.4f}  max={rknn.max():.4f}  mean={rknn.mean():.4f}')

    plot_channel_mean(rknn, 'RKNN rv_feat channel mean/std', os.path.join(args.out, 'rknn_channel_mean.png'))
    plot_feat(rknn, 'RKNN rv_feat (all channels)', os.path.join(args.out, 'rknn_all_channels.png'))

    if args.onnx and os.path.exists(args.onnx):
        onnx = np.load(args.onnx)
        if onnx.ndim == 4:
            onnx = onnx[0]
        print(f'ONNX feat: shape={onnx.shape}  min={onnx.min():.4f}  max={onnx.max():.4f}  mean={onnx.mean():.4f}')
        plot_channel_mean(onnx, 'ONNX rv_feat channel mean/std', os.path.join(args.out, 'onnx_channel_mean.png'))
        plot_diff(rknn, onnx, os.path.join(args.out, 'diff_analysis.png'))
    else:
        print('[INFO] 未提供 ONNX 参考，跳过对比图')

if __name__ == '__main__':
    main()
