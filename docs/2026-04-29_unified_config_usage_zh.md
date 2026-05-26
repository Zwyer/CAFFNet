# 统一配置使用说明（prfnet_semantickitti_unified.yaml）

## 是否会同时读取 pretrain 与 train 配置？

- 不会在一次运行里同时用两套训练参数。
- `train.py` 只读取：`data/model/train/loss/eval/log`
- `tools/pretrain_nomae_pcp.py` 只读取：`data/model/pretrain`
  - 并可选读取 `pretrain_data/pretrain_model/pretrain_log` 覆盖项

## 现在推荐使用的统一配置

- 文件：`prfnet/configs/prfnet_semantickitti_unified.yaml`

## 运行命令

### 1) 预训练（NOMAE+PCP）

```bash
python tools/pretrain_nomae_pcp.py \
  --cfg prfnet/configs/prfnet_semantickitti_unified.yaml
```

### 2) 监督训练（语义分割）

```bash
python train.py \
  --cfg prfnet/configs/prfnet_semantickitti_unified.yaml
```

### 3) 预训练后微调

```bash
python train.py \
  --cfg prfnet/configs/prfnet_semantickitti_unified.yaml \
  --resume <PRETRAIN_CKPT_PATH> \
  --strict-false
```
