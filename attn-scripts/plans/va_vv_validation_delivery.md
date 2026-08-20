# VA/VV 提取验证、性能与交付计划

> 本文覆盖两个计划脚本的统一验证与仓库交付流程。当前不运行这些检查。

## 1. 验证原则

- 先验证数学 helper，再跑最小端到端，再跑正式 10-chunk 任务。
- 每项检查只跑一次；失败后只重跑失败项。
- 不运行与改动无关的 broad suite。
- 不在本地执行 `uv sync`、GPU import 或 CUDA 验证。
- 所有 GPU 验证在代码 commit/push 后，由 `h100-server` 干净 pull 执行。

## 2. 本地静态检查

实施完成后，本地只做：

```text
git status --short --branch
git diff --check
bash -n attn-scripts/run_va_vv_sparsity.sh
bash -n attn-scripts/run_va_vv_attention.sh
python3 -m py_compile <本次修改的 Python 文件>
```

再检查：

- 两个脚本 `--help`。
- 默认输出名和参数互斥校验。
- README 路径、示例和 summary schema 一致。
- 不运行 `uv sync`。
- docs/workflow-only 变更不运行 `bash tests/test.sh`。

## 3. Helper 数值验证（H100）

构造小型随机：

```text
Q video queries
K = video keys + action keys
H heads
D head dim
```

独立 ground truth：

```python
scores = q @ k.T / sqrt(D)
full = torch.softmax(scores, dim=-1)
```

必须验证：

1. recorder shared denominator 输出的 VV slice 等于 `full[..., video_mask]`。
2. VA slice 等于 `full[..., action_mask]`。
3. 每 row `VV.sum + VA.sum` 与 1 的最大误差低于约定阈值。
4. VV/VA slice 各自重归一化后，与 region-only softmax 一致。
5. region-conditional top-p 与 `top_p_sparsity_rows()` 一致。
6. action mask 为空时：mass=0、Sparsity=NaN，不触发 argmax 假成功。
7. 大矩阵路径不存在 in-place overlapping view；row max 必须 clone 或等价安全实现。

边界规则：

- top-p tolerance 继续 `1e-12`。
- 不要求所有 query 逐位相等；0.9 边界允许 ±1 token，但均值差必须符合 `1/N` 上界。

## 4. 端到端 smoke：compact

正式代码推送后，在 H100 可用 GPU 上跑：

```text
chunks=2
step=24
layers=0,14,29
heads=0,1
regions=VV,VA
mode=chunk-sparsity
```

检查目录：

```text
step_024/sparsity/vv/l00.csv
step_024/sparsity/va/l00.csv
step_024/mass/vv/l00.csv
step_024/mass/va/l00.csv
... l14/l29
```

断言：

- 12 个 compact CSV（4 trees × 3 layers）。
- 每个 shape=`2×2`。
- VA Sparsity 第 0 列全 NaN。
- VA mass 第 0 列为 0；VV mass 第 0 列为 1。
- chunk 1 mass_VV+mass_VA≈1。
- summary 的 scheduler timestep 来自实际 step24。
- `joint_stage_verified=true`。
- 无 latents/actions/demo.mp4。

## 5. 端到端 smoke：matrix

同一代码版本跑：

```text
chunks=2
step=24
layers=0,14,29
heads=0,1
regions=VV,VA
mode=matrix
```

断言：

- 每 region 6 个 CSV，共 12 个。
- VV 每文件 `480×480`。
- VA 每文件 `480×32`。
- CSV 无表头。
- 每行字段数固定。
- VV：前 240 rows 只有 240 valid fields，后 240 rows 有 480 valid fields。
- VA：前 240 rows 全空，后 240 rows 有 32 valid fields。
- row0 在 summary 中对应最早 Q token；绘图无需翻转。

数值交叉：

- 抽样至少 3 个 layer/head、两个 chunk 的若干 rows。
- `sum(VV)+sum(VA)≈1`。
- full CSV 重算 compact VA/VV Sparsity 与 compact smoke 一致。

## 6. 向后兼容验证

必须保留旧路径：

1. 运行现有 helper realistic-size compact/legacy test。
2. 旧 `run_vv_sparsity.sh` 小规模输出：
   - 路径仍为 `step_XXX/lXX.csv`。
   - shape、summary 字段和数值不变。
3. `compute_attention_density.py` 对旧 VV matrix 目录仍能读取。

不需要重新跑完整 10 chunks 旧任务，除非具体回归失败。

## 7. 正式 10-chunk compact 验证

推荐在资源足够时使用全部可用、空闲 H100；若只用 0–2：

```text
CUDA_VISIBLE_DEVICES=0,1,2
--gpus 3
--chunks 10
--step 24
--layers all
--heads all
```

验收：

- primary Sparsity CSV：2 regions ×30 layers=60。
- mass CSV：2 regions ×30 layers=60。
- 每个 shape=`24×10`。
- VA 第 0 列 NaN/0 约定正确。
- chunk c 的 N：VV=`240(c+1)`、VA=`32c`。
- `_SUCCESS` 只在所有 rank 完成后出现。
- 记录 wall-clock、GPU 数和输出总大小。

## 8. 正式 full matrix 前置检查

full task 每个 prompt 预计 30–45 GiB CSV 加约 10 GiB 临时 blocks；多 prompt 时按 prompt 数线性放大。启动前：

```text
remote clean git tree
df -h <output filesystem>  # 至少 60 GiB × prompt 数的可用空间
确认输出目标不存在或无 _SUCCESS
确认 GPU 显存余量
```

先用一个 layer、一个 head、10 chunks 估算：

- VV bytes/file。
- VA bytes/file。
- capture 时间、CSV merge 时间。
- 临时 block 峰值。

若线性估算超过可用磁盘，停止正式任务，不自动改成压缩/二进制格式，因为用户明确要求 CSV。

## 9. 正式 10-chunk matrix 验收

目标：

```text
chunks=10
step=24
layers=all
heads=all
```

文件计数：

```text
VV = 720 CSV
VA = 720 CSV
summary.json = 1
_SUCCESS = 1
```

形状：

```text
VV each = 2400×2400
VA each = 2400×288
```

检查策略避免遍历全部数值两遍：

- 对所有文件检查存在、行数和每行字段数。
- 对每个 layer 抽一个 head 做完整数值/NaN pattern 检查。
- 对 layer 0/14/29、head 0/11/23 做 row-sum joint 恒等式。
- 对同样样本复算 top-p 对照 compact 输出。
- 检查 summary 的 token ids、boundaries、actual timestep。

## 10. 性能记录

两个脚本都应打印/写入 summary：

- model load start/end。
- prompt start/end。
- capture elapsed。
- CSV finalize elapsed。
- total wall-clock。
- GPU world size。
- selected layers/heads/steps。
- temporary bytes 与 final bytes（可在 finalize 后统计）。

这样可以区分模型推理时间和大量 CSV I/O 时间。

## 11. Git 交付流程

每轮实现：

1. 本地检查 `git status` 和 `git diff`。
2. 只 stage 本任务文件；不 stage `attn-output`、数据、缓存、agent state 或 sibling `jupyter-plot` 文件。
3. 创建描述性 commit 并 push 当前 branch。
4. 复用一个 `h100-server` SSH/tmux terminal。
5. 远端仓库必须 clean、branch 正确；执行 `git pull --ff-only origin <branch>`。
6. 执行一次 `uv sync --frozen`。
7. 只运行当前阶段需要的 focused GPU check。
8. 若需改代码，回本地修复并重新 commit/push/pull；不直接修改远端 tracked source。
9. 全部完成后关闭复用 terminal。

## 12. Stop conditions

遇到以下情况停止，不覆盖：

- 本地或远端 lingbot-va 出现未知 tracked changes。
- 远端 branch 错误或无法 fast-forward。
- full matrix 预计空间不足 60 GiB。
- action-history 在 chunk 1 以后为空。
- VV+VA row mass 不接近 1。
- compact 与 full CSV 的数学口径无法在容差内对齐。
- 输出目录已有 `_SUCCESS` 且未显式允许 overwrite。
- 3 次修复仍失败时，停止并指出最可能错误的假设。

## 13. 交付完成定义

只有同时满足以下条件才算完成实现：

- 两个新脚本、README 和底层 recorder/server 改动已 commit/push。
- 旧 VV 路径无回归。
- H100 helper、compact smoke、matrix smoke 通过。
- 正式 full run 若用户要求执行，则 1440 CSV、summary、`_SUCCESS` 全部存在并通过抽检。
- 实际耗时和输出大小已记录。
- 没有在远端直接留下未提交源码修改。
