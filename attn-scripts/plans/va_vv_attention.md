# `run_va_vv_attention.sh` 详细计划

> 本文只定义未来 full attention matrix 导出；当前不创建脚本。

## 1. 脚本职责

提取指定 video denoising step 的当前 video Q 对：

- 所有可见 video K（VV）。
- 所有可见 action-history K（VA）。

两个 region 使用同一个 all-visible-K joint softmax denominator。输出用于后续热力图，不在本脚本中绘图或池化。

## 2. 计划 CLI

```bash
uv run bash attn-scripts/run_va_vv_attention.sh \
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
output=attn-output/va-vv-attention-c10-s24
```

`--heads` 对 full CSV 很重要：默认 all 满足正式任务，少量 heads 供 smoke 与磁盘估算。head 子集在 capture 层裁剪；输出文件仍使用原始 head index（例如选择 `0,11,23` 就写 `h00/h11/h23`），不重新编号。summary 的 `head_indices` 必须与临时 block 第一维和实际文件集合一致。

## 3. Server 模式映射

计划使用：

```text
--attention-analysis
--attention-analysis-only
--attention-export-mode matrix
--attention-regions VV,VA
--attention-layers ...
--attention-heads ...
--attention-timesteps ...
--attention-output ...
```

只记录 video loop；action loop query 会被 recorder 忽略，但其最终 cache K/V 会成为下一 chunk 的 VA history。

## 4. 概率语义

每个 CSV 数值均为：

```text
exp(q·k/sqrt(d)) /
Σ exp(q·k_all/sqrt(d))
```

分母覆盖该 query 当前可见的全部 video+action keys。

禁止：

- 分别对 VV/VA 做 softmax 后把结果当 raw attention。
- 为了让各 region 行和为 1 而写回重归一化值。
- 用 zeros 填充 future/unavailable keys。

full CSV 的目的之一就是看 VA/VV 在 joint softmax 中的真实质量竞争。

## 5. 输出树和文件数

```text
<output>/<prompt-slug>/
├── summary.json
├── _SUCCESS
└── step_024/
    ├── vv/
    │   ├── l00h00.csv
    │   ├── ...
    │   └── l29h23.csv
    └── va/
        ├── l00h00.csv
        ├── ...
        └── l29h23.csv
```

默认：

```text
VV files = 30 × 24 = 720
VA files = 30 × 24 = 720
Total matrix CSVs = 1440
```

## 6. 行轴：current video Q

两个 region 使用完全相同的 row 轴。

### 默认 logical row shape

```text
10 chunks × 240 current-video Q tokens = 2400 rows
```

顺序：

1. chunk index 升序 `0..9`。
2. chunk 内 frame id 升序。
3. frame 内 spatial token index 升序。

CSV 第 0 行就是热力图最上方；row index 从上向下增加。绘图应：

```python
imshow(matrix, origin="upper", ...)
```

不得 `flipud`，不得使用 `origin="lower"`。

`summary.json` 记录：

- `row_token_ids`
- `row_chunk_boundaries = [240, 480, ..., 2160]`
- `row_axis_direction = "top-to-bottom increasing"`

## 7. 列轴：region-specific K history

### 7.1 VV

最终 logical shape：

```text
2400 rows × 2400 columns
```

chunk `c` 的 240 行只在前：

```text
240 × (c+1)
```

列有数值，其余 future columns 为空。

VV 包含当前 chunk video keys，这是 self-attention 的真实可见集合。

### 7.2 VA

最终 logical shape：

```text
2400 rows × 288 columns
```

chunk `c` 的 240 行只在前：

```text
32 × c
```

列有数值：

- chunk 0：全行为空字段。
- chunk 1：前 32 列有效。
- ...
- chunk 9：前 288 列有效。

VA 不含“当前 chunk action”，因为当前 chunk 的 action loop 在 video loop 之后执行。

### 7.3 空字段

CSV 必须保留固定 logical width：

- 已可见 key：写 `%.9g` 数字。
- 尚不可见 key：写 empty field。
- 行末仍保留足够逗号，使每行列数固定。

这样 `np.genfromtxt(..., filling_values=np.nan)` 可直接生成供热力图使用的矩阵。

## 8. Token metadata

summary 的每个 region 记录：

```json
{
  "csv_directory": "step_024/vv",
  "csv_pattern": "lXXhXX.csv",
  "logical_shape_per_csv": [2400, 2400],
  "history_token_ids": [[0, 0], [0, 1]],
  "history_chunk_boundaries": [240, 480]
}
```

VA token id：

```text
(frame_id, action_slot_within_frame)
```

VV token id：

```text
(frame_id, spatial_index_within_frame)
```

两个 region 共享 row token ids，但不能共享 K column metadata。

## 9. Block 存储与 CSV 写入

### 9.1 Capture 阶段

每个 layer/chunk 生成一个临时 `.pt` payload：

```python
{
  "attention": {
    "VV": Tensor[heads,Q,K_v],
    "VA": Tensor[heads,Q,K_a]
  }
}
```

- tensor 在保存前 `.cpu()`。
- chunk 0 VA tensor shape `[H,240,0]`。
- worker/layer/chunk 路径沿用 `.attention-blocks/worker_XX/...`。

### 9.2 Save 阶段

- 一次只处理一个 layer，避免把 30 层全部读入内存。
- 每个 selected head 同时或连续写 VA/VV `.csv.tmp`。
- 写完 `flush/close` 后原子 rename。
- 对 chronological prefix 走现有 fast path；保留 cache eviction/reordering 的 general fallback。
- 成功后删除该 worker 的临时 blocks。

## 10. 体积与性能预算

默认 10 chunks、30 layers、24 heads、step 24：

- VV logical fields：约 4.15 billion。
- VV actual numeric fields：约 2.28 billion。
- VA logical fields：约 0.50 billion。
- VA actual numeric fields：约 0.25 billion。
- 临时 fp32 blocks：约 10 GiB 数量级。
- 最终 text CSV：每个 prompt 预计约 30–45 GiB，依赖数字位数和 empty fields；多 prompt 线性倍增。

运行前必须：

1. 检查输出盘至少 60 GiB 可用。
2. 先跑少量 layers/heads 估算本机实际 bytes/head。
3. 默认只提取 step 24；除非明确请求，不扩到多 step。
4. 不在 GPU 上长期保留多个 chunk 的矩阵。
5. 不生成重复的 combined joint CSV，以免再增加数十 GiB。

## 11. 热力图消费契约

绘图侧可直接：

```python
matrix = np.genfromtxt(
    csv_path,
    delimiter=",",
    dtype=np.float32,
    filling_values=np.nan,
    ndmin=2,
)
```

约定：

- rows=Q，columns=K history。
- `origin="upper"`。
- NaN/future cells 使用独立背景色。
- VA 与 VV 若要直接比较，应使用共同的 raw-probability normalization 或明确各自 color scale；不要先行 region 重归一化。
- chunk 边界从 summary 读取，不从固定 240/32 硬编码，以便其他 config 可复用。

## 12. 数值验证

对抽样 layer/head/chunk/query：

1. 从 VV/VA CSV 取同一 row 的非空值。
2. 验证 `sum(vv)+sum(va)≈1`。
3. 分别重归一化 region，离线复算 top-p Sparsity。
4. 与 compact script 的对应 head/chunk 值比较。
5. 校验：

```text
VV valid columns = 240 × (chunk+1)
VA valid columns = 32 × chunk
```

6. chunk 0 VA 全空；chunk 1 以后 VA 非空。
7. 最后一行/列 token id 与 summary 一致。

## 13. 最小 smoke

计划先跑：

```text
chunks=2
step=24
layers=0,14,29
heads=0,1
gpus=3
```

预期每个 region 6 个 CSV：

- VV shape `480×480`。
- VA shape `480×32`。
- VV 前 240 rows：前 240 columns numeric、后 240 empty。
- VV 后 240 rows：480 columns numeric。
- VA 前 240 rows：全 empty。
- VA 后 240 rows：32 columns numeric。

## 14. 完成标准

- 每个 region/layer/head 恰好一个无表头 CSV。
- 10 chunks/step24/all layers/all heads 的 1440 文件全部生成。
- rows/columns/token ordering 和 summary 一致。
- top-to-bottom Q 方向明确且无需绘图侧翻转。
- raw joint probability 恒等式通过。
- 输出完成前不创建 `_SUCCESS`。
- analysis-only 模式不产生其他推理产物。
