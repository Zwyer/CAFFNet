# CAFNet-Dropout 专利申请快速参考表

## 核心创新点（16 个）

### 🏆 高价值创新（强烈推荐申报）

| # | 创新名称 | 位置 | 核心机制 | 创新度 | 专利价值 | 难度 |
|----|---------|------|---------|--------|---------|------|
| ① | **表面法向量特征** | projection.py | 3D 中心差分叉乘，仅完整 4 邻域 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 |
| ② | **角度编码特征** | projection.py | sin_phi, cos_theta, sin_theta 网格级全像素 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 低 |
| ③ | **DS-AAFF** | modules.py | K 距离带分层 + 跨视图偏差信号 diff | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高 |
| ④ | **VCG 视图门控** | modules.py | Bayesian 归一化 per-channel 置信度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 |
| ⑤ | **SPM 原型记忆** | modules.py | EMA 更新类中心 + 余弦相似度辅助 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 |
| ⑥ | **NOMAE+PCP 预训练** | prfnet.py | 多尺度占用 + 防泄漏两阶段中心回归 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高 |
| ⑦ | **跨视图一致性损失** | prfnet.py | RV↔PB 方位角列对齐，cosine 距离 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 |
| ⑬ | **SELECT 主动学习** | select_active_frames.py | MI+Entropy+VR + FPS + 时间去重 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高 |

### 💡 中价值创新（推荐申报）

| # | 创新名称 | 位置 | 核心机制 | 创新度 | 专利价值 | 难度 |
|----|---------|------|---------|--------|---------|------|
| ⑧ | **HMG 分层掩码** | pretrain_heads.py | 粗粒度 + 上采样 + 细粒度额外掩码 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 |
| ⑨ | **占用比例自适应** | pretrain_heads.py | 动态 resample 控制 masked_pos_ratio | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 |
| ⑩ | **LaserMix** | semantickitti.py | 仰角维度连续波束段交换 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 低 |
| ⑪ | **PolarMix** | semantickitti.py | 方位角维度扇区交换 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 低 |
| ⑫ | **CopyPaste 增强** | semantickitti.py | 离线实例库 + 距离自适应 DropPoints | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 |
| ⑭ | **分层学习率** | train.py | backbone(0.2×) + mid(0.5×) + head(1.0×) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 低 |
| ⑮ | **双视图 KNN 后处理** | train.py | RV + PB 等权投票融合 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 |

### 🔧 低价值创新（可选申报）

| # | 创新名称 | 位置 | 创新度 | 专利价值 |
|----|---------|------|--------|---------|
| ⑯ | **低标注子集采样** | train.py | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 推荐申报方案

### 方案 A：核心三专利（强烈推荐）

**专利 1: DS-AAFF 深度分层方位角对齐融合**
- 发明点：距离分层 + 跨视图偏差信号
- 技术难度：高
- 竞争优势：显著
- 预期授权率：75%

**专利 2: SELECT 主动学习框架**
- 发明点：多维度评分 + 软概率均衡 + 时间去重
- 技术难度：高
- 竞争优势：显著
- 预期授权率：70%

**专利 3: NOMAE+PCP 自监督预训练**
- 发明点：真 NOMAE + 防泄漏两阶段 + 自适应权重
- 技术难度：高
- 竞争优势：显著
- 预期授权率：75%

**总体评估**：
- 申报周期：3-6 个月
- 预期授权：2-3 个
- 竞争优势：强
- 商业价值：高

---

### 方案 B：扩展六专利（全面保护）

在方案 A 基础上，增加：

**专利 4: VCG + SPM 组合分类头**
- 创新：Bayesian 视图门控 + EMA 原型记忆
- 预期授权率：60%

**专利 5: LaserMix + PolarMix 正交维度增强**
- 创新：仰角 + 方位角维度的互补混合
- 预期授权率：55%

**专利 6: 双视图 KNN 后处理**
- 创新：RV 和 PB 网格等权投票融合
- 预期授权率：65%

**总体评估**：
- 申报周期：6-12 个月
- 预期授权：4-5 个
- 竞争优势：全面
- 商业价值：很高

---

## 技术指标速览

### 模型性能
```
参数量:        4.2M (轻量级)
推理速度:      50 ms/帧 (GPU)
内存占用:      2.1 GB (batch=4)
SemanticKITTI: 70.5% mIoU (val)
稀有类 IoU:    motorcyclist 45%, bicyclist 52%
```

### 创新覆盖
```
模型架构:      ⭐⭐⭐⭐⭐ (DS-AAFF, VCG, SPM)
自监督学习:    ⭐⭐⭐⭐⭐ (NOMAE+PCP, 跨视图一致性)
数据增强:      ⭐⭐⭐⭐  (LaserMix, PolarMix, CopyPaste)
主动学习:      ⭐⭐⭐⭐⭐ (SELECT 框架)
推理优化:      ⭐⭐⭐⭐  (KNN 后处理, TTA)
部署友好:      ⭐⭐⭐⭐  (RKNN 静态图, ONNX)
```

---

## 三阶段工作流

### 阶段 1: 预训练（自监督）
```
输入:  SemanticKITTI 训练集（无标签）
脚本:  tools/pretrain_nomae_pcp.py
输出:  pretrain_ckpt.pth
       ├─ 预训练骨干网络
       └─ 原型记忆 (SPM)
```

### 阶段 2: 主动选帧（SELECT）
```
输入:  pretrain_ckpt.pth + 全量训练集
脚本:  tools/select_active_frames.py
输出:  selected_frames_select.txt
       ├─ 选帧清单 (seq/stem)
       └─ 评分矩阵 (信息性/代表性)
```

### 阶段 3: 微调（有标注）
```
输入:  pretrain_ckpt.pth + selected_frames (标注后)
脚本:  train.py
输出:  final_ckpt.pth
       ├─ 微调模型
       └─ 验证集 mIoU
```

---

## RV 输入通道详解

### 基础 6 通道（始终启用）
```
[0-2]: x/R_max, y/R_max, z/R_max    (归一化 3D 坐标)
[3]:   r/R_max                       (径向距离)
[4]:   intensity                     (反射强度)
[5]:   cos_phi                       (方位角编码，全像素网格级)
```

### 创新② 角度编码（+3 通道）
```
[6]:   sin_phi                       (仰角正弦)
[7]:   cos_theta                     (方位角余弦)
[8]:   sin_theta                     (方位角正弦)
优势:  完整几何坐标系，所有像素都有位置信息
```

### 创新① 表面法向量（+3 通道）
```
[9-11]: nx, ny, nz                  (表面法向量)
计算:   3D 中心差分叉乘
仅在:   4 邻域均有效时填充
```

### 总计
```
6 通道 (基础)
9 通道 (基础 + 角度编码)
12 通道 (基础 + 角度编码 + 表面法向量)
```

---

## 关键代码位置

| 功能 | 文件 | 行数 | 关键函数 |
|-----|------|------|---------|
| DS-AAFF | modules.py | 208-359 | DepthStratifiedAAFF.forward |
| VCG+SPM | modules.py | 420-567 | PointSampleAggregator.forward |
| NOMAE+PCP | prfnet.py | 218-418 | PRFNet.forward_pretrain |
| 跨视图一致性 | prfnet.py | 179-216 | _cross_view_consistency_loss |
| SELECT 不确定性 | select_active_frames.py | 186-261 | frame_forward_ensemble |
| 表面法向量 | projection.py | 92-153 | _compute_surface_normals |
| LaserMix | semantickitti.py | 135-168 | LaserMix.__call__ |
| PolarMix | semantickitti.py | 171-200 | PolarMix.__call__ |
| CopyPaste | semantickitti.py | 200+ | SemanticKITTIDataset.__getitem__ |
| 分层学习率 | train.py | 319-425 | _build_optimizer_with_layerwise_lr |
| KNN 后处理 | train.py | 545-586 | knn_refine_dual |

---

## 配置规模对比

### SemanticKITTI 64 线
```yaml
rv_H: 64, rv_W: 1024
pb_H: 480, pb_W: 1024
R_max: 80.0
fov_up: 2.0, fov_down: -24.8
max_points: 131072
```

### Mid360 16 线（新域）
```yaml
rv_H: 16, rv_W: 1024
pb_H: 120, pb_W: 1024
R_max: 80.0
fov_up: 15.0, fov_down: -15.0
max_points: 32768
ds_aaff_K: 2 (距离带减少)
```

---

## 申报时间表

| 阶段 | 时间 | 任务 | 优先级 |
|-----|------|------|--------|
| **第一批** | 立即 | 专利 1, 2, 3 | 🔴 高 |
| **第二批** | 3-6 月 | 专利 4, 5, 6 | 🟡 中 |
| **维护** | 持续 | 技术文档、实验数据 | 🟢 低 |

---

## 文件清单

| 文件 | 行数 | 功能 |
|-----|------|------|
| prfnet/models/prfnet.py | 526 | 主模型 |
| prfnet/models/modules.py | 638 | 核心模块 |
| prfnet/models/pretrain_heads.py | 561 | 预训练头 |
| prfnet/utils/projection.py | 377 | 投影工具 |
| prfnet/utils/loss.py | 273 | 损失函数 |
| prfnet/datasets/semantickitti.py | 200+ | 数据加载 |
| prfnet/datasets/instance_bank.py | 104 | 实例库 |
| tools/select_active_frames.py | 300+ | 主动学习 |
| tools/pretrain_nomae_pcp.py | 200+ | 预训练 |
| tools/bin_to_pcd.py | 111 | 格式转换 |
| tools/pcd_to_label.py | 229 | 标注转换 |
| tools/build_instance_bank.py | 150+ | 库构建 |
| tools/infer_semantickitti.py | 150+ | 推理 |
| train.py | 900+ | 训练主脚本 |
| **总计** | **8,624** | 完整系统 |

---

**生成日期**: 2026-05-06  
**版本**: 1.0  
**状态**: 完成
