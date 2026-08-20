# `run_va_vv_sparsity.sh` 详细计划

> 本文只定义未来脚本与实现契约；当前不创建脚本。

## 1. 脚本职责

在一次 i2va 推理中，对指定 video step、每个 selected layer/head/chunk 同时输出：

1. VV region-conditional top-p=0.9 Sparsity。
2. VA region-conditional top-p=0.9 Sparsity。
3. VV raw joint-softmax region mass。
4. VA raw joint-softmax region mass。

不生成 full attention matrices，不画图，不保存 latents/actions，不做 VAE decode。

## 2. 计划 CLI

```bash
uv run bash attn-scripts/run_va_vv_sparsity.sh \
  --model MODEL_DIR \
  (--prompts FILE | --prompt TEXT) \
  [--chunks N] \
  [--step LIST] \
  [--layers all|LIST] \
  [--heads all|LIST] \
  [--gpus N] \
  [--seed N] \
  [--chunk-tokens N] \
  [--config robotwin_i2av] \
  [--output DIR] \
  [--overwrite]
```

默认：

```text
chunks=10
step=24
layers=all
heads=all
gpus=8
seed=0
chunk-tokens=512
config=robotwin_i2av
output=attn-output/va-vv-sparsity-c10-s24
```

参数行为沿用 `run_vv_sparsity.sh`：

- `--prompts` 与 `--prompt` 二选一。
- 多 prompt 单次模型加载，逐 prompt 重置相同 seed 和初始 observation latent。
- `--layers`、`--heads` 支持逗号列表，便于最小 H100 smoke。
- `--overwrite` 默认关闭；已存在 `_SUCCESS` 时拒绝覆盖。

## 3. 调用 server 的计划参数

shell 最终映射到类似：

```text
--attention-analysis
--attention-analysis-only
--attention-export-mode chunk-sparsity
--attention-regions VV,VA
--attention-layers ...
--attention-heads ...
--attention-timesteps ...
--attention-output ...
--attention-chunk-tokens ...
--attention-seed ...
```

旧 `--attention-vv-chunk-sparsity` 保留，仅供旧脚本使用。

## 4. 数学输出

### 4.1 Sparsity CSV

对 region `R`：

```text
N_R = 当前 query 可见的 region key 数
p_R(k|R) = raw_joint_probability / region_mass
k = 累计 p_R 达到 0.9 所需最少 token 数
S_R = 1 - k/N_R
```

每个 CSV cell：

```text
mean over current-video queries in that chunk of S_R(q)
```

### 4.2 Mass CSV

每个 CSV cell：

```text
mean over current-video queries in that chunk of
Σ_{k in R} p_joint(k)
```

逐 head/chunk：

```text
mass_vv + mass_va ≈ 1
```

### 4.3 chunk 0

- `sparsity/va/lXX.csv` 的第 0 列写 `nan`。
- `mass/va/lXX.csv` 的第 0 列写 `0`。
- `sparsity/vv` 正常计算。
- `mass/vv` 应为 1（浮点容差内）。

## 5. 输出树

```text
<output>/<prompt-slug>/
├── summary.json
├── _SUCCESS
└── step_024/
    ├── sparsity/
    │   ├── vv/
    │   │   ├── l00.csv
    │   │   └── ... l29.csv
    │   └── va/
    │       ├── l00.csv
    │       └── ... l29.csv
    └── mass/
        ├── vv/
        │   ├── l00.csv
        │   └── ... l29.csv
        └── va/
            ├── l00.csv
            └── ... l29.csv
```

### CSV 格式

- 无表头。
- 行按 `head_indices` 升序；row position 不等于原始 head id 时，以 summary 的 `head_indices` 为准。
- head 子集在 capture 层裁剪，block 第一维和 CSV 行数都必须等于 `len(head_indices)`。
- 列按 `chunk_indices` 升序。
- 默认 `24×10`。
- 数字格式 `%.9g`。
- `nan` 只允许出现在 VA 的空-support chunk；summary 声明位置。
- 不把 `nan` 写成 0。

## 6. Summary schema 重点

建议新增：

```json
{
  "attention_query_region": "V",
  "attention_key_regions": ["VV", "VA"],
  "softmax_scope": "all visible video and action keys",
  "sparsity_scope": "region-conditional",
  "sparsity_top_p": 0.9,
  "sparsity_formula": "1 - k / N_region",
  "mass_formula": "sum of raw joint-softmax probabilities in region",
  "empty_region_policy": {
    "VA_chunk_0_sparsity": "nan",
    "VA_chunk_0_mass": 0.0
  },
  "csv_header": false,
  "head_indices": [0, 1, 2, 3],
  "layer_indices": [0, 14, 29]
}
```

每个 step summary 记录：

- `csv_shape_per_layer`
- `num_chunks`
- `num_heads`
- `chunk_indices`
- 每个 chunk 的 `N_VV`、`N_VA`、all key count
- mass 校验的最大 row/head 聚合误差
- `joint_stage_verified=true`

## 7. GPU 计算计划

每个 layer/chunk：

1. 选当前 video Q。
2. streaming 扫描 all keys，得到 joint denominator。
3. 依次处理 VV、VA：
   - 计算 region score matrix。
   - 计算 raw region mass。
   - region 内稳定 softmax、sort、fp64 cumulative。
   - 立即 reduce 到 `[H]` Sparsity 和 `[H]` mean mass。
4. 释放 score/sort 中间 tensor。
5. 仅在 save 时将小向量搬到 CPU 并写 CSV。

相较 full matrix，此模式不保存 `.pt` attention blocks；按先释放 shared score、再顺序处理 VV/VA region 的实现，预计额外显存峰值低于约 300 MiB/正在处理的 layer，并且不会随 chunk 数累积 GPU tensor。正式值必须以 H100 smoke 的 `max_memory_allocated` 实测记录为准。

## 8. 与旧 VV 输出的关系

必须做双向回归：

- 新 `sparsity/vv/lXX.csv` 与旧 `run_vv_sparsity.sh` 的 `lXX.csv` 逐 cell 比较。
- 以 fp32 score / fp64 cumulative 的数值容差为准，不要求逐位一致；0.9 边界只允许 ±1 token，compact 均值偏差应不超过对应 `1/N_region` 的边界量级。
- 旧脚本输出路径和 summary 不改变。

## 9. 离线复算

计划扩展 `script/compute_attention_density.py` 或新增共享 reader，使其支持：

```bash
--region vv|va
--summary <summary.json>
```

读取 full matrix 时：

- 只对指定 region 文件的非空字段计入 N。
- 先按 region row mass 重归一化，再应用同一 top-p helper。
- VA chunk 0 返回 NaN，不参与错误的 0 值平均。

compact 与 full CSV 可按 head/chunk 对齐复算，不只比较 layer 总均值。

## 10. 最小 smoke 验证

建议先跑：

```text
chunks=2
step=24
layers=0,14,29
heads=0,1
gpus=3
```

预期：

- 每个 metric/region 有 3 个 layer CSV。
- 每个 CSV shape=`2×2`。
- VA Sparsity 第 0 列全是 `nan`。
- VA mass 第 0 列全是 0。
- chunk 1 `N_VV=480`、`N_VA=32`。
- 每个 head/chunk `mass_vv+mass_va≈1`。
- VV compact 与旧路径一致。

## 11. 完成标准

- 一个推理进程组同时给出 VA/VV，而不是分别加载模型跑两次。
- 数学定义、chunk0 空 region、mass 与 Sparsity 区分写入 README/summary。
- 默认 10 chunks、step 24、all layers/heads 输出稳定。
- 不产生 latents/actions/demo.mp4。
- 多 prompt 仍只加载一次模型。
- 旧 VV-only workflow 无回归。
