# 近两年点云语义分割三大方向检索：主流方案与端侧部署潜力（截至 2026-04-27）

## 0. 检索范围与判定标准

- 检索时间窗：`2024-04-27` 到 `2026-04-27`
- 主来源：CVPR/ICCV OpenAccess、arXiv、OpenReview（优先一手论文页）
- 任务范围：
  1. 大规模无标注点云 + 100-200 帧标注点云训练语义分割
  2. 图像 + 点云多模态语义分割
  3. 开放词汇点云语义分割
- 端侧潜力分级（基于论文公开信息做工程推断）：
  - `高`：推理阶段可 LiDAR-only、可离线预计算文本/图像原型、不依赖在线重型 VLM
  - `中`：推理有多帧或轻量跨模态开销，但可裁剪或蒸馏
  - `低`：推理依赖重型多模态/生成模型或长时序大内存

## 1. 大规模无标注 + 100-200 帧标注（低标注/半监督）

### 1.1 近两年主流路线

- 半监督 Teacher-Student + 伪标签质量提升（2024-2026 的主线）
- 时序一致性建模（多帧 LiDAR 的高/低时序敏感特征）
- 主动学习/冷启动优化（在极低标注预算下优先选“信息量高”的样本）
- 跨模态预训练后小样本微调（利用 2D 基座模型降低 3D 标注依赖）

### 1.2 代表方法与端侧潜力

| 方法 | 时间 | 核心点 | 与“100-200 帧标注”关联 | 端侧潜力 |
|---|---|---|---|---|
| DDSemi | CVPR 2024 | 密度引导对比学习 + 双空间困难样本采样 | 半监督设定，面向少标注+大量无标注 | 中 |
| AIScene | CVPR 2025 | 场景内/场景间亲和性，点擦除减少伪标签噪声传播 | 报告 1% 标注下显著增益 | 中-高 |
| HiLoTs | CVPR 2025 | 利用长时序高/低敏感特征并 Teacher-Student 对齐 | 面向低标注自动驾驶序列 | 中 |
| RePL | arXiv/OpenReview 2026 | 伪标签纠错（masked reconstruction）+ 理论分析 | SOTA 指向低标注半监督场景 | 中-高 |
| BaSAL | ICRA 2024 | 面向主动学习冷启动和类不平衡的 size-balanced 采样 | 5% 标注接近全监督，适合“先采样再标注” | 高（训练策略） |
| BALViT | ICRA 2025 Workshop | 冻结视觉基座 + 2D-3D adapter 做标签高效学习 | 明确针对小数据 regime | 中 |
| PSA-SSL | CVPR 2025 | 自监督预训练保留 pose/size 几何信息 | 声称可用更少标注达到更强下游分割 | 高（预训练后推理轻） |

### 1.3 小结（这一主题）

- 若你手里是“海量无标注 + 100-200 帧标注”，当前最稳妥的是：
  1. 先做 `PSA-SSL/BALViT` 类预训练或初始化
  2. 再上 `AIScene/RePL` 类半监督伪标签优化
  3. 如果标注预算仍可控，前置 `BaSAL` 类主动学习做标注帧选择

## 2. 图像 + 点云多模态语义分割

### 2.1 近两年主流路线

- 时序多模态聚合（历史点云 + 历史图像）
- 跨模态记忆与对齐（统一 3D 空间位置编码）
- 自动化伪标注数据引擎（减少人工标注成本）
- 训练期跨模态增强，推理期尽量降成本（蒸馏或零额外开销）

### 2.2 代表方法与端侧潜力

| 方法 | 时间 | 核心点 | 部署特征 | 端侧潜力 |
|---|---|---|---|---|
| TASeg | CVPR 2024 | 时序 LiDAR 聚合 + 时序图像融合 + 蒸馏 | 准确率强，但多帧多模态链路重 | 中-低 |
| LiMA | ICCV 2025 | 图像到 LiDAR 的长时序记忆聚合预训练 | 论文强调下游“无额外计算开销” | 高 |
| SAM4D | ICCV 2025 | Camera/LiDAR 流统一 promptable 分割基础模型 | 能力强但体系较重，偏云边协同 | 低-中 |
| SAL-4D | CVPR 2025 | 多模态桥接 + VOS/VLM 蒸馏到 4D LiDAR | 训练链路重，推理可转向 LiDAR 模型 | 中 |
| 3D-AVS | CVPR 2025 | 图像+点云联合识别并自动生成词表再分割 | 开放词表能力强，在线多模态开销需控制 | 中 |

### 2.3 小结（这一主题）

- 面向端侧，最有价值的不是“全程多模态在线推理”，而是：
  - `训练时多模态增强/蒸馏` + `推理时 LiDAR-only`
- 这一点和你当前“训练用图像语义、推理去重型分支”的方向一致。

## 3. 开放词汇点云语义分割

### 3.1 近两年主流路线

- 2D 开放词汇/视觉语言模型向 3D 蒸馏（AFOV、LOSC）
- 用扩散模型或生成模型增强开放语义表示（Diff2Scene、IGLOSS）
- 课程学习与伪标签精炼（PGOV3D、LOSC）
- 强化推理效率，减少在线 VLM 依赖（FOLK 的思路在工程上很关键）

### 3.2 代表方法与端侧潜力

| 方法 | 时间 | 核心点 | 部署特征 | 端侧潜力 |
|---|---|---|---|---|
| AFOV | arXiv 2024/2025 | 2D 开放词汇模型蒸馏到 3D，含注释自由学习 | 可走“训练重、推理轻”路径 | 高 |
| Diff2Scene | ECCV 2024 | 用 text-to-image diffusion 表示做 OV 3D 分割 | 语义泛化强，但生成式链路偏重 | 低-中 |
| PGOV3D | arXiv 2025 | Partial-to-Global 课程学习利用跨视角语义 | 精度导向，部署要看学生网络大小 | 中 |
| LOSC | arXiv 2025/2026v2 | 时空一致性 + 图像增强鲁棒性整合伪标签 | 训练过程复杂，推理可转 3D 单网 | 中-高 |
| 3D-AVS | CVPR 2025 | 自动词表生成 + 3D 分割 | 开放词汇自动化强，在线多模态要裁剪 | 中 |
| IGLOSS | arXiv 2026 | 通过文本生成原型图像再与 3D 特征匹配 | 若原型离线缓存，推理可简化 | 中-高 |
| FOLK* | arXiv 2025 | 知识蒸馏减少 2D 映射/VLM 推理负担，强调快 | 虽是 instance segmentation，但“快推理范式”可迁移到语义分割 | 高 |

`*` FOLK 属于开放词汇 3D 实例分割，不是严格语义分割，但其“蒸馏后快推理”范式对端侧语义分割很有参考价值。

### 3.3 小结（这一主题）

- 开放词汇 3D 的工程分水岭：是否把重型 VLM 依赖留在训练期。
- 若目标是端侧，优先选择“离线蒸馏 + 在线相似度匹配”的路线，而非在线多模态大模型。

## 4. 面向你项目的优先级建议（可直接落地）

1. 低标注训练主线：`PSA-SSL/BALViT 初始化 -> AIScene/RePL 半监督优化`
2. 多模态主线：训练阶段借鉴 `TASeg/LiMA` 的时序跨模态聚合，推理阶段保留 LiDAR-only
3. 开放词汇主线：以 `AFOV/LOSC/IGLOSS` 为主参考，吸收 `FOLK` 的高效蒸馏思想
4. 端侧指标建议同步汇报：`mIoU`、`unseen mIoU`、`FPS`、`显存`、`模型参数量`

## 5. 参考链接（按主题分组）

### 5.1 低标注/半监督

- DDSemi (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Li_Density-Guided_Semi-Supervised_3D_Semantic_Segmentation_with_Dual-Space_Hardness_Sampling_CVPR_2024_paper.html
- AIScene (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Exploring_Scene_Affinity_for_Semi-Supervised_LiDAR_Semantic_Segmentation_CVPR_2025_paper.html
- HiLoTs (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Lin_HiLoTs_High-Low_Temporal_Sensitive_Representation_Learning_for_Semi-Supervised_LiDAR_Segmentation_CVPR_2025_paper.html
- RePL (arXiv 2026): https://arxiv.org/abs/2604.06825
- RePL (OpenReview ICLR 2026 submission): https://openreview.net/forum?id=zS1bPtMlt9
- BaSAL (IEEE): https://ieeexplore.ieee.org/document/10611122/
- BaSAL (OpenReview ICRA 2024): https://openreview.net/forum?id=a3tSH3YuBc
- BALViT (OpenReview ICRA 2025 FMNS): https://openreview.net/forum?id=w3MTdtHYKY
- PSA-SSL (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Nisar_PSA-SSL_Pose_and_Size-aware_Self-Supervised_Learning_on_LiDAR_Point_Clouds_CVPR_2025_paper.html

### 5.2 图像+点云多模态

- TASeg (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Wu_TASeg_Temporal_Aggregation_Network_for_LiDAR_Semantic_Segmentation_CVPR_2024_paper.html
- LiMA (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Beyond_One_Shot_Beyond_One_Perspective_Cross-View_and_Long-Horizon_Distillation_ICCV_2025_paper.html
- SAM4D (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Xu_SAM4D_Segment_Anything_in_Camera_and_LiDAR_Streams_ICCV_2025_paper.html
- SAM4D (arXiv): https://arxiv.org/abs/2506.21547
- SAL-4D (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Zero-Shot_4D_Lidar_Panoptic_Segmentation_CVPR_2025_paper.html

### 5.3 开放词汇点云语义分割

- AFOV (arXiv): https://arxiv.org/abs/2405.15286
- Diff2Scene (ECCV 2024 / arXiv): https://arxiv.org/abs/2407.13642
- PGOV3D (arXiv): https://arxiv.org/abs/2506.23607
- LOSC (arXiv): https://arxiv.org/abs/2507.07605
- 3D-AVS (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Wei_3D-AVS_LiDAR-based_3D_Auto-Vocabulary_Segmentation_CVPR_2025_paper.html
- IGLOSS (arXiv): https://arxiv.org/abs/2604.01361
- FOLK (arXiv): https://arxiv.org/abs/2510.08849
- OA-CNNs (效率参考，CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Peng_OA-CNNs_Omni-Adaptive_Sparse_CNNs_for_3D_Semantic_Segmentation_CVPR_2024_paper.html

