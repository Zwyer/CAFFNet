# 预训练效果可视化（无需标签）

预训练阶段没有 GT 也可以用三类信号验证模型学到了东西：① 重建图 ② 特征 PCA 着色 ③ 无监督聚类。`tools/visualize_pretrain.py` 把前两类组合到一个脚本里，输出 `.ply` 与 `.png`，不依赖 X11。

## 1. 输出说明

每帧产出 4 个文件（`<seq>_<stem>_*`）：

| 文件 | 含义 | 怎么看 |
|---|---|---|
| `*_pca.ply` | 3D 点云，颜色 = encoder+decoder 逐点融合特征做 PCA→RGB | 用 CloudCompare / Meshlab 打开。**同一物体颜色相近、不同物体明显分块** = 预训练学到了语义结构。颜色噪点状均匀分布 = 表征没学进去。 |
| `*_pca_bev.png` | 上面那个 PCA 颜色直接做 BEV 俯视图 | 服务器端快速预览，不用下载 ply。 |
| `*_mae_rv.png` | RV 四联图：occupancy GT / mask / 可见输入 / 预测 occupancy | 第 4 行（pred）在 mask 内的高亮区域应贴近第 1 行（target）的形状。完全空白或满图亮表示重建头未训练好。 |
| `*_mae_pb.png` | PB（Polar BEV）同上 | 同样判读方法。 |

> **PCA 颜色没有固定语义**：第 1 帧"车=红"不代表第 2 帧"车=红"。同帧内的相对一致性才是关键。

## 2. 运行

```bash
# 1) 找到预训练 ckpt（最新一次预训练会写到 runs/prfnet_16lidar_pretrain_rankme/<时间戳>/checkpoints/）
ls runs/prfnet_16lidar_pretrain_rankme/*/checkpoints/

# 2) 运行可视化（默认对 selected_frames_select.txt 列出的所有帧出图）
python tools/visualize_pretrain.py \
    --cfg  prfnet/configs/prfnet_16lidar_unified.yaml \
    --ckpt runs/prfnet_16lidar_pretrain_rankme/<TS>/checkpoints/best_rankme.pth \
    --list selected_frames_select.txt \
    --out_dir runs/viz_pretrain \
    --max_frames 20          # 0 = 全部；建议先 5~10 帧看效果
```

输出落到 `runs/viz_pretrain/`。

## 3. 常用参数

| 参数 | 默认 | 用途 |
|---|---|---|
| `--max_frames` | 0 (全部) | 先小批快速看；线上 `selected_frames_select.txt` 300 帧太多 |
| `--rv_mask_ratio` / `--pb_mask_ratio` | 从 cfg 读 | 想看更激进 / 更稀疏的掩码下重建效果可以手动指定 |
| `--seed` | 0 | 固定掩码 RNG，不同 ckpt 对比时用同一掩码更公平 |
| `--occ_scale_idx` | 0 | NOMAE 是多尺度 occ 预测，0 = 最细尺度，常看这个 |
| `--skip_ply` | off | 只要 png 时加上，省磁盘 |

## 4. 怎么判断"预训练有效"

| 现象 | 解释 |
|---|---|
| PCA BEV 上路面整体一种色、车一种色、树一种色 | ✅ encoder 学到了几何 / 语义聚类 |
| PCA 全图随机彩噪、无空间结构 | ❌ 表征塌缩或没收敛，看 RankMe 是否一直 ≤ 数据维度的 1/2 |
| MAE 预测 occ 在 mask 区还原了 target 的形状（>50% 重叠） | ✅ NOMAE 收敛到非平凡解 |
| MAE 预测全图均匀 0.5 或纯黑 | ❌ occ 头没学会，检查 `loss_occ` 是否还在显著下降，pos_weight 是否太大/小 |

## 5. 后续可扩展（可选）

- **KMeans 无监督聚类着色**：把 `extract_pretrain_point_features` 的输出过一遍 sklearn KMeans(n_clusters=7~10) 直接换颜色映射；视觉上最接近"伪分割图"，但簇号要肉眼对应类别。
- **跨 ckpt 对比**：固定 `--seed` 跑两个 ckpt（如 ep000 vs ep050），对同一帧的 `*_pca_bev.png` 并排放就是"训练前 / 训练后"的视觉对照。
- **TensorBoard add_mesh**：把 PCA RGB 写到 TB 里可直接 3D 旋转看；当前脚本未启用，需要时把 `xyz` 与 `rgb` 写入 `SummaryWriter.add_mesh`。
