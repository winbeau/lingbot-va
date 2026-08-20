# 共享 recorder 与联合 softmax 核心计划

> 本文是 `../PLAN_VA_VV_EXTRACTION.md` 的模块子计划。当前只描述实现，不修改代码。

## 1. 当前基线

### 1.1 Hook 位置

`WanAttention.forward()` 中：

1. 生成并 rotary 处理 query/key。
2. 将当前 key/value 写入 KV cache。
3. 从 cache 取出所有有效 video/action keys。
4. 调用 `JointAttentionRecorder.capture(...)`。
5. 调用真实 `attn_op(query, key, value)`。

因此 recorder 拿到的是与真实 attention 同一份 rotary 后 query/key，以及同一可见 KV 集合。

### 1.2 当前 recorder 的限制

- query mask 固定 `query_token_types == 0`，即当前 video Q。
- key slice 固定 `key_token_types == 0`，即只物化 VV。
- full-matrix 模式使用 `compute_token_probability_slice()`：
  - Pass 1 对所有 visible video+action keys 计算 streaming softmax denominator。
  - Pass 2 只物化 VV。
- compact 模式使用 `compute_token_sparsity_slice()`：
  - 直接在 VV logits 上做 region-only softmax。
  - 因为 full denominator 在 VV 重归一化后消去，所以与 legacy CSV 复算等价。

## 2. 目标内部配置

将 recorder 初始化参数从专用 bool 泛化为：

```python
export_mode: Literal["matrix", "chunk_sparsity"]
regions: tuple[Literal["VV", "VA"], ...]
head_indices: list[int] | None
```

约束：

- query region 当前固定 V。
- `regions` 默认由入口脚本设为 `("VV", "VA")`。
- `head_indices=None` 表示全部 heads。
- head 子集必须在 capture 层生效，而不是先计算 24 个 head 再在 save 阶段丢弃：query/key score、sort 和临时 block 都只保留 `head_indices`；因此 block 的第一维是 `len(head_indices)`，summary 的 `head_indices` 与 CSV 行数必须严格一致。默认 `all` 仍保留 24 heads，旧 VV-only 模式不启用子集裁剪。
- 旧参数 `vv_chunk_sparsity=True` 继续映射到：

```python
export_mode="chunk_sparsity"
regions=("VV",)
```

这样不改变 `run_vv_sparsity.sh`。

## 3. 联合 softmax共享计算

### 3.1 基本定义

令选中的当前 video query 为 `Q`，所有可见 keys 为 `K_all`：

```text
scores_all = Q @ K_all^T / sqrt(D)
Z = Σ exp(scores_all)
```

region masks：

```text
VV: key_token_types == 0
VA: key_token_types == 1
```

raw region probability：

```text
P_R = exp(scores_R) / Z
```

### 3.2 推荐 helper 分层

建议把当前函数拆成三个小层次：

```python
compute_streaming_softmax_stats(...)
# -> row_max, row_sumexp

materialize_probability_slice(..., row_max, row_sumexp, key_mask)
# -> [H, Q, K_R] raw joint probability

compute_region_metrics(..., region_masks)
# -> region sparsity, mean mass, key counts；可选 raw slices
```

兼容 wrappers：

```python
compute_token_probability_slice(...)
compute_token_sparsity_slice(...)
```

继续保留原函数签名和现有测试口径，内部调用新 helper。

### 3.3 内存控制

- denominator 继续按 `attention_chunk_tokens` 分块扫描所有 keys。
- VV 与 VA slice 顺序物化，不同时保留两套大 score/sort tensor。
- full matrix 模式：每个 region slice 计算后立即 `.cpu()` 并写临时 `.pt` block。
- compact 模式：每个 region 只保留 `[H]` Sparsity 和 `[H]` mean mass。
- 保留 fp32 score matmul；top-p cumulative 使用 fp64。

## 4. Sparsity 与 mass 的共享算法

对 region `R`：

```text
mass_R(q) = sum(P_R(q, :))
conditional_R = P_R / mass_R
```

当 `N_R > 0`：

1. 对 conditional probability 降序排序。
2. 稳定 tie-break：概率相同按原 key column index 升序。
3. 找到累计到 `TOP_P - TOP_P_TOLERANCE` 的最小 `k`。
4. `S_R(q) = 1 - k/N_R`。
5. 对 Q 求均值，得到 `[H]`。

实现上可继续利用 denominator 消去：

```text
softmax(scores_R) == P_R / mass_R
```

但 mass 必须由 shared joint denominator 计算，不能从 region-only softmax 推断。

空 region：

- `N_R=0`。
- `mass_R=0`。
- `S_R=nan`。
- 不调用 sort/argmax。

## 5. Token 顺序和身份

### 5.1 Query rows

所有 region 共用当前 video query token ids：

```text
(frame_id, spatial_index_within_frame)
```

默认每 frame 120 tokens，每 chunk 两帧，共 240 query rows；10 chunks 拼为 2400 rows。

### 5.2 VV columns

video history token ids：

```text
(frame_id, spatial_index_within_frame)
```

按 cache 中首次出现顺序取 ordered union。10 chunks 无 eviction 时是严格时序前缀，最终 2400 columns。

### 5.3 VA columns

action history token ids：

```text
(frame_id, action_slot_within_frame)
```

`token_frame_ids` 已在 model 中 floor 为整数 frame id；同一 frame 内按 cache 中 token 顺序编号 `0..15`。最终只覆盖前 9 个 action chunks，共 18 frames、288 columns。

### 5.4 Metadata

每个 chunk 至少记录：

```json
{
  "chunk_index": 0,
  "step_index": 24,
  "scheduler_timestep": 172.41378784179688,
  "query_tokens": 240,
  "visible_video_history_tokens": 240,
  "visible_action_history_tokens": 0,
  "all_key_tokens_in_softmax": 240,
  "query_frame_start": 0,
  "query_frame_end": 2
}
```

额外在 step summary 中记录：

- `row_token_ids`
- `row_chunk_boundaries`
- `regions.VV.history_token_ids`
- `regions.VV.history_chunk_boundaries`
- `regions.VA.history_token_ids`
- `regions.VA.history_chunk_boundaries`
- 每个 region 的 logical shape、CSV pattern、empty-field 语义

## 6. Recorder 状态结构

建议将 `block_files` 从：

```python
layer -> [block]
```

泛化为：

```python
step -> layer -> [chunk payload]
```

payload 视模式而定：

### Matrix payload

```python
{
    "VV": Path(...chunk.pt),
    "VA": Path(...chunk.pt),
}
```

或一个 `.pt`：

```python
{
    "attention": {
        "VV": Tensor[H,Q,K_v],
        "VA": Tensor[H,Q,K_a],
    }
}
```

推荐一个 `.pt`，共享 metadata，减少 inode 数；chunk 0 的 VA tensor 允许 shape `[H,Q,0]`。

**向后兼容要求**：旧 legacy full-matrix 模式的 payload 仍必须保持 `{"attention": Tensor[H,Q,K_v]}` 形状与旧 writer 契约；只有新双 region mode 才允许 `{"attention": {"VV": Tensor, "VA": Tensor}}`。`_num_heads()`、`_save_layer_csvs()` 和旧 CSV 输出必须按 payload 类型分支，不能让新 dict 结构泄漏到旧路径。

### Compact payload

```python
{
    "sparsity": {"VV": Tensor[H], "VA": Tensor[H]},
    "mass": {"VV": Tensor[H], "VA": Tensor[H]},
}
```

## 7. 分布式完成语义

当前 rank 0 可能在其他 rank 的大 CSV 尚未写完时先生成 `summary.json`。full matrix 体积扩大后要修正：

1. 每个 rank 写自己分配的 layer CSV。
2. 每个 rank 完成原子 rename 和 block cleanup。
3. `torch.distributed.barrier()`。
4. rank 0 写 `summary.json.tmp`，原子 rename 为 `summary.json`。
5. rank 0 创建空文件 `_SUCCESS`。
6. 再 barrier，随后清 cache/进入下一个 prompt。

绘图或 rsync 只认存在 `_SUCCESS` 的实验目录。

## 8. Backward compatibility

必须保留：

- `--attention-vv-chunk-sparsity`
- 现有 VV-only `step_XXX/lXX.csv`
- 现有 full VV `step_XXX/lXXhYY.csv`
- 现有 summary 关键字段
- Phase A 必须额外回归 legacy full-matrix payload、writer 和 CSV 读取，不得只验证 compact 路径
- `compute_attention_density.py` 对旧目录的读取

新模式采用新目录结构，不重解释旧文件：

- 旧 mode：原路径不变。
- 新 VA/VV mode：显式 `sparsity/{region}`、`mass/{region}` 或 `{region}/lXXhYY.csv`。

## 9. 错误校验

capture/save 阶段应拒绝：

- query 中没有 video token。
- VV key support 为空。
- chunk 1 以后 VA support 为空。
- `video_count + action_count != all_key_count`。
- 各 layer 捕获 chunk 数不一致。
- 各 layer 的 query/key count、token ids、step/timestep 不一致。
- requested heads 越界或不同 layer 的 head 数不一致。
- 输出目标已有 `_SUCCESS`，除非显式 `--overwrite`。

允许：

- chunk 0 VA support 为空。
- rank 在 layer 数小于 world size 时没有任务。

## 10. 共享核心完成标准

- 一次 joint denominator 可供 VA/VV 共用。
- raw slice 恒等式通过：`sum(VV)+sum(VA)≈1`。
- region 重归一化后与直接 region logits softmax 一致。
- 旧 VV compact 和 legacy 数值不回归。
- matrix 和 compact 共用同一 token/order metadata。
- summary 只在所有 rank 完成后发布。
