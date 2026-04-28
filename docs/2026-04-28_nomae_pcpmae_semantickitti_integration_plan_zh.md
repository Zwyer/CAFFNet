# NOMAE + PCP-MAE 融合 PRFNet（SemanticKITTI）详细修改计划（2026-04-28）

## 目标

在不增加推理开销的前提下，将 `NOMAE + PCP-MAE` 融入当前 PRFNet：

1. 新增训练期自监督预训练路径（不影响现有监督训练入口）
2. 预训练完成后可直接热启动 `train.py` 做语义分割微调
3. 推理阶段保持现有 `export_forward` 路径不变

## 总体方案

采用两阶段训练：

1. 阶段 A（自监督预训练）  
   仅训练编码解码主干 + NOMAE/PCP 头，优化占据重建与中心预测损失。
2. 阶段 B（语义分割微调）  
   使用阶段 A 权重热启动现有监督训练（CE + Lovasz + Aux）。

## 代码改动清单

1. `prfnet/models/pretrain_heads.py`  
   - 新增 `NOMAEPCPPretrainHead`
   - 新增 `build_random_mask_like`

2. `prfnet/models/prfnet.py`  
   - 新增 `self.pretrain_head_rv` / `self.pretrain_head_pb`
   - 新增 `forward_pretrain(...)`

3. `prfnet/utils/loss.py`  
   - 新增 `NOMAEPCPLoss`

4. `tools/pretrain_nomae_pcp.py`  
   - 新增预训练脚本（独立于 `train.py`）
   - 支持 `--dry-run` 一步冒烟测试

5. `prfnet/configs/pretrain_nomae_pcp_semantickitti.yaml`  
   - 新增预训练配置

## 损失定义（当前最小实现）

- RV 占据目标：`rv_img[ch=3] > 0`
- PB 占据目标：`pb_img[ch=8]`
- RV 中心目标：`rv_img[ch=0:3]`
- PB 中心目标：`pb_img[ch=0:3]`

总损失：

`L = lambda_occ * (L_occ_rv + L_occ_pb) + lambda_pcp * (L_pcp_rv + L_pcp_pb)`

## Dry Test 计划

1. `python -m py_compile` 做语法检查
2. `python tools/pretrain_nomae_pcp.py --dry-run ...` 做 1 step 前向/反向冒烟
3. 若无数据集环境，仅执行前两项中的语法检查并记录说明

## 后续增强（下一迭代）

1. 将随机掩码升级为 NOMAE 的多尺度邻域掩码策略
2. 将 PCP 目标从中心 proxy 升级为 patch-level 几何中心预测
3. 加入预训练 checkpoint 到监督训练的自动加载入口
