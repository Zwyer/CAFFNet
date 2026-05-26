# 当前代码检查、优化建议与训练启动指南（SemanticKITTI，2026-04-29）

## 1. 代码检查结论（先给结论）

- 当前主训练链路（`train.py`）整体可用，结构清晰。
- 但存在 **2 个高优先级问题**（推理路径存在运行时错误风险）。
- `NOMAE+PCP` 预训练脚本已具备最小可跑能力（含 dry-run 回退逻辑），可作为第一版验证。

---

## 2. 发现的问题（按严重级别）

### 严重（High）

1. `infer_single_bin` 中 `PolarBEVProjector` 构造参数与实现不匹配，调用会报错。  
   - 调用位置：`tools/infer_semantickitti.py:1191-1194`  
   - 当前调用使用 `H=..., W=..., z_min=..., z_max=...`。  
   - 实际构造函数是 `PolarBEVProjector(H_p, W_p, R_max)`：`prfnet/utils/projection.py:257`。  
   - 影响：`infer_single_bin(...)` 直接运行会触发 `TypeError`。

2. `_compute_geo_features` 调用了不存在的 `compute_pixel_coords`。  
   - 调用位置：`tools/infer_semantickitti.py:138`  
   - `RangeImageProjector` 仅有 `compute_sample_coords`，无 `compute_pixel_coords`。  
   - 影响：当启用几何点特征路径时，会触发 `AttributeError`。

### 中等（Medium）

1. `tools/pretrain_nomae_pcp.py` 中 `GradScaler` 固定 device 字符串为 `"cuda"`。  
   - 位置：`tools/pretrain_nomae_pcp.py:152`  
   - 虽然 CPU 下会自动降级并可运行，但会有无意义 warning；建议与 `autocast` 一样按 `device.type` 控制。

2. 预训练脚本里 `pretrain.scheduler` 配置字段未实际分支使用。  
   - 配置中有 `scheduler: cosine`，脚本当前固定使用 `CosineAnnealingLR`。  
   - 建议：要么删配置字段，要么实现 `cosine/onecycle` 分支，避免“配置可写但无效”。

### 轻微（Low）

1. 仓库是新初始化并一次性 root commit，历史不可追溯到原 CAFFNet 上游。  
   - 对你现在开发不影响；但后续若要与上游同步，建议在新仓里补充 README 与分支策略说明。

---

## 3. 建议优先优化项（执行顺序）

1. 先修复 `infer_single_bin` 的 `PolarBEVProjector` 参数名（`H_p/W_p`，去掉 `z_min/z_max`）。
2. 在 `RangeImageProjector` 中补 `compute_pixel_coords`，或把 `_compute_geo_features` 改为由 `compute_sample_coords` 反算像素行列。
3. 统一 AMP 逻辑：`GradScaler(device.type, enabled=amp_enabled)`。
4. 给预训练脚本补 “resume/保存 best/日志文件” 三项，以便正式跑实验。
5. 增加一个 `docs/quickstart` 的最小命令集（防止后续命令散落在注释里）。

### 可改进清单（NOMAE/PCP 对齐优先）

1. 将当前随机像素 mask 升级为 NOMAE 风格的“多尺度邻域占据建模”mask 与目标构造。  
2. 将当前 PCP 像素中心 proxy 回归升级为 PCP-MAE 风格的 patch/center 设计（避免信息泄漏、对齐 patch 级目标）。  
3. 对上述两项分别做单独消融与组合消融：  
   - Baseline（当前实现）  
   - +NOMAE 多尺度邻域  
   - +PCP patch/center  
   - +NOMAE +PCP  
4. 在低标注设定（如 1%/5%）同步报告 `mIoU + 训练时长 + 显存占用`，明确收益/成本比。  

---

## 4. 如何开始训练（建议流程）

这里按你当前代码结构，给出可直接执行的流程。

### 阶段 A：准备环境

建议 Python 环境需包含：
- `python>=3.10`
- `torch`
- `numpy`
- `pyyaml`
- `tqdm`
- `tensorboard`

若你使用现有环境（示例）：
```bash
conda activate py311_torch240_cu118
```

### 阶段 B：准备 SemanticKITTI 数据

期望目录结构（与 `SemanticKITTIDataset` 一致）：
```text
<DATA_ROOT>/
  sequences/
    00/
      velodyne/*.bin
      labels/*.label
    01/
    ...
    21/
```

然后在配置中设置：
- `prfnet/configs/prfnet_semantickitti.yaml` 的 `data.root`
- `prfnet/configs/pretrain_nomae_pcp_semantickitti.yaml` 的 `data.root`

### 阶段 C：可选构建 Copy-Paste 实例库（建议）

如果你使用监督训练配置里 `copy_paste.enable: true`，需要先构建实例库：
```bash
python tools/build_instance_bank.py \
  --root <DATA_ROOT> \
  --out instance_bank_semantickitti.npz \
  --num-workers 8
```

### 阶段 D：先做预训练（NOMAE+PCP）

```bash
python tools/pretrain_nomae_pcp.py \
  --cfg prfnet/configs/pretrain_nomae_pcp_semantickitti.yaml
```

输出目录：
- `runs/prfnet_pretrain_nomae_pcp/<timestamp>/checkpoints/latest.pth`

### 阶段 E：监督微调（语义分割）

```bash
python train.py \
  --cfg prfnet/configs/prfnet_semantickitti.yaml \
  --resume <PRETRAIN_CKPT_PATH> \
  --strict-false
```

说明：
- `--strict-false` 是必须的（预训练头与分割头键集合不完全一致）。

### 阶段 F：验证/推理

val mIoU：
```bash
python tools/infer_semantickitti.py \
  --cfg prfnet/configs/prfnet_semantickitti.yaml \
  --ckpt <SEG_BEST_CKPT> \
  --split val
```

test 预测导出：
```bash
python tools/infer_semantickitti.py \
  --cfg prfnet/configs/prfnet_semantickitti.yaml \
  --ckpt <SEG_BEST_CKPT> \
  --split test \
  --output_dir predictions/
```

---

## 5. 训练所需数据清单

### 必需

1. SemanticKITTI 原始数据：
   - 点云：`sequences/*/velodyne/*.bin`
   - 语义标签（训练/验证）：`sequences/*/labels/*.label`

2. 配置文件中的正确路径：
   - `data.root` 指向上述数据根目录

### 可选但强烈建议

1. Copy-Paste 实例库：`instance_bank_semantickitti.npz`  
   - 用于长尾类增强（motorcyclist/bicyclist/person 等）

2. 预训练 checkpoint：
   - 用于热启动监督训练，提高低标注/小样本稳定性

---

## 6. 你现在可以直接执行的最小启动步骤

1. 修改两个配置文件的 `data.root` 为你的 SemanticKITTI 路径。
2. （可选）执行 `build_instance_bank.py` 生成实例库。
3. 跑 `tools/pretrain_nomae_pcp.py` 预训练。
4. 跑 `train.py --resume ... --strict-false` 微调。
5. 用 `tools/infer_semantickitti.py --split val` 看 mIoU。
