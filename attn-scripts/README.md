# Attention Scripts

统一存放注意力提取与分析入口。脚本名保持短、可搜索：

- `run_<target>_<metric>.sh`：端到端推理与提取。
- `compute_<metric>.py`：离线计算。
- `inspect_<artifact>.py`：格式或正确性检查。

## Script Index

| Script | Purpose |
|---|---|
| `run_vv_sparsity.sh` | 提取第二阶段 i2va 的 VV top-p Sparsity；每个 layer 输出一个 head × chunk CSV。 |

## VV Sparsity Quick Start

准备一个 prompt 文件，每行一个提示词；空行和以 `#` 开头的行会被忽略：

```text
Grab the white mug and place it on the rack.
Move the white mug to the left side of the table.
```

在 `h100-server` 上运行：

```bash
cd /root/wenbiao_zhao/lingbot-va && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run bash attn-scripts/run_vv_sparsity.sh \
  --model /root/wenbiao_zhao/models/lingbot-va-posttrain-robotwin \
  --prompts prompts.txt \
  --num-prompts 2 \
  --chunks 10 \
  --step 24 \
  --layers all \
  --gpus 8
```

常用参数：

- `--chunks N`：chunk 数，至少 `2`，默认 `10`。
- `--step LIST`：video denoising step；支持单值或逗号列表，默认 `24`。
- `--layers LIST`：`all` 或层号列表，例如 `0,14,29`；默认 `all`。
- `--output DIR`：输出根目录；默认按 chunk 和 step 自动命名。
- `--seed N`：每个 prompt 重置相同随机种子，默认 `0`。
- `--prompt TEXT`：只跑一个 prompt，和 `--prompts FILE` 二选一。
- `--num-prompts N`：使用 `--prompts FILE` 时只跑过滤后的前 `N` 条 prompt；空行和注释行不计数。省略时跑完整文件。

脚本在一个进程组内依次运行全部 prompt，复用模型加载和初始 observation latent；提取模式不会保存中间 latents/actions，也不会解码 `demo.mp4`。

## Output

默认输出示例：

```text
attn-output/vv-sparsity-c10-s24/
└── <prompt-slug>/
    ├── summary.json
    └── step_024/
        ├── l00.csv
        ├── ...
        └── l29.csv
```

每个 `lXX.csv`：

- 无表头。
- 行是 head，按 `0..23` 排列。
- 列是 chunk，按 `0..N-1` 排列。
- 10 chunks 时每行正好 10 个数、9 个逗号。
- 每个值采用现有 top-p=0.9 定义：对当前 chunk 的每个 video query，在可见 video keys 上求达到 0.9 累积注意力所需最少 token 数 `k`，计算 `1-k/N`，再对 query 求均值。

## Correctness Notes

- 只允许 `robotwin_i2av` 等 i2va 路径；chunk 1 以后必须能看到 action-history keys，否则导出失败。
- 默认 step `24` 是最后一次真正更新 video latent 的 forward；step `25` 是 cache-only。实际 shifted scheduler timestep 始终以 `summary.json` 为准。
- chunk 0 尚无前序 action history，但仍保留为第 0 列；后续 chunk 会经过联合 video/action cache 校验。
- compact 路径直接由 VV logits 在 GPU 上计算 video-only 条件 softmax；这与从全模态 softmax 中取 VV 后再次对 video keys 归一化在数学上等价。由于 compact 路径保留 fp64 累积而矩阵乘是 fp32，极少数恰好落在 0.9 边界的 query 可能出现 1 token 的数值级差异。
