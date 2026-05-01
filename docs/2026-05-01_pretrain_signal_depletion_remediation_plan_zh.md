# 预训练信号耗尽问题整改计划（NOMAE + PCP，SemanticKITTI）

## 1. 问题定义（当前现象）

你观察到：

- `occ/pcp` 很快降到 `1e-4` 级别；
- `gnorm` 长时间接近 `0`；
- 后续大量 step 收益很小。

这通常意味着当前 pretext task 对模型过易，核心风险是：

1. mask 仅作用在 loss，不作用在输入（信息泄漏）；
2. OCC 在全空域监督，模型“预测全空”也能拿到很低 loss；
3. PCP 目标设计存在捷径，语义表示学习不足；
4. 训练没有“信号耗尽早停”机制，计算浪费。

---

## 2. 目标（按优先级）

1. 阻止无效计算：预训练在信号耗尽时自动早停。  
2. 提升任务难度：mask 比例更高 + curriculum。  
3. 消除泄漏捷径：输入级 masking + PCP anti-leakage。  
4. 让 OCC 学到“结构”：多尺度邻域占据目标 + 类别不平衡损失。  
5. 保证可诊断：日志/TensorBoard 记录关键统计，支持快速定位问题。

---

## 3. 分阶段实施计划

## Phase A（当天可完成）：止损与难度提升

### A1. 早停（signal depletion early stop）

**改动位置**

- `tools/pretrain_nomae_pcp.py`

**实现方案**

- 新增滑动窗口均值指标（推荐 window=`200` steps）：
  - `occ_ma`, `pcp_ma`, `gnorm_ma`
- 早停条件（满足即触发）：
  - `occ_ma < 1e-3`
  - `pcp_ma < 1e-3`
  - `gnorm_ma < 0.05`
  - 持续 `patience_steps`（推荐 `400~800`）  
- 触发后保存 `early_stop.pth` 并结束训练。

**配置项（新增到 `pretrain`）**

- `early_stop_enable: true`
- `early_stop_window: 200`
- `early_stop_occ_thr: 1.0e-3`
- `early_stop_pcp_thr: 1.0e-3`
- `early_stop_gnorm_thr: 0.05`
- `early_stop_patience_steps: 600`

**验收标准**

- `pretrain.log` 明确打印触发原因与 step；
- TensorBoard 能看到触发前 MA 曲线收敛。

---

### A2. 更高 mask + curriculum

**改动位置**

- `tools/pretrain_nomae_pcp.py`（调度逻辑）
- `prfnet/models/prfnet.py`（接收每 step 的动态 mask ratio）
- `prfnet/configs/prfnet_semantickitti_unified.yaml`

**实现方案**

- 基础目标：最终 mask 提升到 `0.90~0.92`；
- curriculum：
  - 前 `20%` step 固定 `0.70`；
  - 后 `80%` step 从 `0.70` 线性升到 `0.92`；
- RV/PB 分别可配（必要时 PB 稍高）。

**配置项**

- `mask_curriculum_enable: true`
- `mask_curriculum_warmup_ratio: 0.20`
- `rv_mask_ratio_start: 0.70`
- `rv_mask_ratio_end: 0.92`
- `pb_mask_ratio_start: 0.70`
- `pb_mask_ratio_end: 0.92`

**验收标准**

- TB 出现 `pretrain/rv_mask_ratio_step`、`pretrain/pb_mask_ratio_step` 上升曲线；
- 与固定 0.70 相比，`gnorm` 不再过早贴近 0。

---

## Phase B（核心改造）：输入级 masking + OCC 改造

### B1. 输入级 masking（堵住“只在 loss 上 mask”的捷径）

**改动位置**

- `prfnet/models/prfnet.py::forward_pretrain`
- `prfnet/models/pretrain_heads.py`（mask 生成器可复用）

**实现方案**

- 将 mask 应用于 `rv_img/pb_img` 输入，而非只用于 loss：
  - masked 区域置零（先做 zero-mask 版本）；
  - 后续可升级为 learnable `mask_token`；
- 编码器看到的是“缺失输入”，不是完整输入。

**细节建议**

- 可只遮挡几何主通道（`xyz/range/occ`），保留少量辅助通道，避免数值崩；
- 对 RV 和 PB 使用独立 mask，减少跨视图共轭泄漏。

**验收标准**

- 同等训练步数下，`occ/pcp` 下降速度明显变慢；
- `gnorm` 不再在单 epoch 内快速衰减至 0。

---

### B2. OCC 改造为“多尺度邻域占据建模”

**改动位置**

- `prfnet/models/pretrain_heads.py`
- `prfnet/models/prfnet.py`
- `prfnet/utils/loss.py`（可新增 occ 专用损失类）

**实现方案**

- 从当前“单尺度像素占据重建”升级为多尺度邻域目标：
  - 以 occupancy map 为 base；
  - 通过 `max_pool` 构造尺度 `s in {1,2,4}` 的邻域占据标签；
  - 在 masked 区域监督多尺度 logits；
- 监督区域限制为“masked 且邻域内有结构信息”的位置，避免全空域主导。

**关键点**

- 避免“全预测空”低损失：  
  - 只监督 informative masked cells（非空邻域优先）；  
  - 或采用正负样本均衡采样。

**验收标准**

- `masked_positive_ratio` 保持在可学习区间（建议 `5%~40%`）；
- OCC loss 不再在极短时间内跌入 `1e-4`。

---

## Phase C（核心改造）：PCP anti-leakage

### C1. PCP 两阶段目标（先预测，再替换）

**改动位置**

- `prfnet/models/pretrain_heads.py`
- `prfnet/models/prfnet.py::forward_pretrain`

**实现方案**

- 当前是直接回归中心 proxy，改为两阶段：
  1. `center_predictor` 先对 masked 位置预测 center；
  2. decoder/重建分支使用“预测 center 替代真实 center”（可 `detach`）；
- 目标是消除用真实 center 走捷径的通路。

**训练细节**

- 推荐先用 `stop-grad` 替换（稳定）；
- 后续可试小权重联合训练，避免梯度互相污染。

**验收标准**

- PCP loss 收敛速度放缓但更稳定；
- 下游微调（100/200 帧）mIoU 提升优于旧版 PCP。

---

## Phase D（损失与采样稳态化）

### D1. OCC 不平衡损失

**改动位置**

- `prfnet/utils/loss.py::NOMAEPCPLoss`（或新增 `OCCBalancedLoss`）

**实现方案**

- OCC 使用 `BCEWithLogits(pos_weight=...)` 或 focal BCE；
- `pos_weight` 可来自：
  - 全局先验（推荐先验常量）；
  - 或 batch 统计（带 EMA 平滑）。

**推荐初值**

- `occ_pos_weight: 5.0`（再按曲线调到 `3~10`）

---

### D2. 提高 PCP 权重

**配置建议**

- `lambda_pcp`: `0.5 -> 1.0`（首选），必要时到 `2.0`；
- 保持 `lambda_occ=1.0` 先不动，先观察任务平衡。

---

### D3. 约束 masked 正样本比例

**改动位置**

- `prfnet/models/pretrain_heads.py`（mask 构造）
- `tools/pretrain_nomae_pcp.py`（统计与报警）

**实现方案**

- 在 mask 采样时引入占据感知：
  - 保证 masked 区域正样本比例不低于阈值（如 `min=0.08`）；
  - 上限可设（如 `max=0.50`）防止全难样本不稳定；
- 若不满足，重采样 mask（限制重采样次数）。

**验收标准**

- TB 记录 `diag/masked_positive_ratio_step`；
- 比例长期落在设定区间。

---

## 4. 配置草案（建议新增字段）

建议在 `prfnet/configs/prfnet_semantickitti_unified.yaml` 的 `pretrain` 下加入：

```yaml
pretrain:
  # existing
  epochs: 10
  lr: 8.0e-4
  rv_mask_ratio: 0.70
  pb_mask_ratio: 0.70
  lambda_occ: 1.0
  lambda_pcp: 1.0

  # early stop
  early_stop_enable: true
  early_stop_window: 200
  early_stop_occ_thr: 1.0e-3
  early_stop_pcp_thr: 1.0e-3
  early_stop_gnorm_thr: 0.05
  early_stop_patience_steps: 600

  # mask curriculum
  mask_curriculum_enable: true
  mask_curriculum_warmup_ratio: 0.20
  rv_mask_ratio_start: 0.70
  rv_mask_ratio_end: 0.92
  pb_mask_ratio_start: 0.70
  pb_mask_ratio_end: 0.92

  # leakage control
  input_masking_enable: true
  input_masking_mode: zero      # zero | token
  pcp_stopgrad_replace: true

  # occ balance
  occ_loss_type: bce_pos_weight # bce | bce_pos_weight | focal
  occ_pos_weight: 5.0

  # masked positive ratio control
  mask_pos_ratio_control_enable: true
  mask_pos_ratio_min: 0.08
  mask_pos_ratio_max: 0.50
  mask_resample_max_tries: 5
```

---

## 5. 日志与 TensorBoard 增强项（必须）

在现有日志基础上新增：

1. `pretrain/occ_ma`, `pretrain/pcp_ma`, `diag/gnorm_ma`  
2. `diag/masked_positive_ratio_step`  
3. `diag/mask_resample_count_step`  
4. `diag/early_stop_counter`  
5. `pretrain/rv_mask_ratio_step`, `pretrain/pb_mask_ratio_step`（curriculum 已有则保留）

并在 `pretrain.log` 明确记录：

- 早停阈值；
- 触发 step；
- 触发时 MA 数值快照。

---

## 6. 实验与验收（建议最小消融矩阵）

在同一数据与随机种子下，至少做：

1. Baseline（当前实现）  
2. + Phase A（早停 + 高 mask + curriculum）  
3. + Phase B（输入级 mask + 多尺度 OCC）  
4. + Phase C（PCP anti-leakage）  
5. + Phase D（pos_weight + mask 正样本比例约束）

统一比较：

- 预训练阶段：`occ/pcp` 曲线形态、`gnorm`、有效训练步数；
- 微调阶段（100/200 帧）：`val mIoU`、收敛 epoch、稳定性（3 seeds 方差）。

---

## 7. 风险与回滚策略

主要风险：

1. 难度提升过快导致不收敛；  
2. mask 正样本约束过强导致采样偏差；  
3. PCP anti-leakage 设计不当造成梯度不稳定。

回滚顺序（从激进到保守）：

1. 降低 `*_mask_ratio_end` 到 `0.85`；  
2. 关闭 `mask_pos_ratio_control_enable`；  
3. 保留输入级 mask，但先关闭 PCP 替换分支；  
4. 仅保留 Phase A，确认基础收益后再逐项加回。

---

## 8. 预计收益

短期（Phase A）：

- 明显减少“无效后半程训练”；
- 训练资源利用率提升。

中期（Phase B/C/D）：

- 预训练任务难度与信息含量提升；
- 下游低标注微调（100/200 帧）更可能获得稳定增益；
- 与 NOMAE/PCP-MAE 核心思想对齐度更高，创新性与说服力更强。

