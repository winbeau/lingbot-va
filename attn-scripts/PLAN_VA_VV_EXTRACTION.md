# VA/VV 联合注意力提取实施计划

> 状态：**实现已落地，待 H100 focused validation**
> 目标仓库：`lingbot-va`
> 计划位置：`attn-scripts/`
> 基线：现有 `run_vv_sparsity.sh`、`JointAttentionRecorder`、`wan_va_server`、VV compact/legacy 导出路径

## 1. 目标

后续实现两个统一命名、职责清晰的入口：

| 计划脚本 | 目的 | 默认输出 |
|---|---|---|
| `attn-scripts/run_va_vv_sparsity.sh` | 在第二阶段 i2va 的同一次 video forward 中，同时计算 VA 与 VV 的 top-p=0.9 Sparsity，并记录两类 region 在联合 softmax 中的注意力质量 | 每个 region/layer 一个 `head × chunk` 无表头 CSV |
| `attn-scripts/run_va_vv_attention.sh` | 提取 video step 24、10 chunks、全部 layer/head 的 VA 与 VV 原始联合-softmax注意力切片，供热力图使用 | 每个 region/layer/head 一个无表头 CSV |

这里的 region 固定为：

- **VV**：video query → video key。
- **VA**：video query → action-history key。
- 不包含 AV、AA；query 始终来自 video denoising forward 的当前 video chunk。

## 2. 不变约束

1. 只分析 `robotwin_i2av` 等 i2va 路径的第二阶段联合 video/action cache。
2. 默认 `chunks=10`、video step index=`24`、layers=`all`、heads=`all`。
3. step 24 是第 25 次真实 video latent 更新；step 25 是 scheduler timestep 0 的 cache-only forward。实际 shifted timestep 必须读取 `summary.json`。
4. 当前路径是 LingBot-VA 1.0 风格 velocity FlowMatch，不按 VA2 distilled student 解释。
5. 保留现有 `run_vv_sparsity.sh` 的 CLI、输出和数学语义，不能破坏已经验证过的 VV 工作流。
6. 当前阶段只落盘计划；不新增 shell 脚本、不改 Python 实现、不运行 GPU 提取。

## 3. 已确认的数据流事实

- `WanAttention.forward()` 先把当前 K/V 临时写入 cache，再调用 recorder，因此：
  - VV 在 chunk `c` 可见 `240 × (c+1)` 个 video keys，包含当前 chunk。
  - VA 在 chunk `c` 可见 `32 × c` 个 action-history keys，只包含之前 chunk 的 action。
- 10 chunks、step 24 的默认 RoboTwin 形状：
  - 每个 chunk 当前 video query：240 tokens。
  - 所有 query 拼接：2400 rows。
  - VV history union：2400 columns。
  - VA history union：288 columns（最后一个 action chunk 在最后一次 video forward 之后才生成，因此不进入本次 VA heatmap）。
- 每个 chunk 的联合 key 数应满足：

```text
all_keys(c) = 240 × (c + 1) + 32 × c
```

- chunk 0 的 VA support 为空；chunk 1–9 必须存在 action-history keys，否则视为误入纯视频阶段或 cache 语义错误。

## 4. 数学口径（冻结后再实现）

### 4.1 联合 softmax

对每个 layer/head/query，联合分布为：

```text
p_joint(k) = exp(score_k) / Σ_{all visible video+action keys} exp(score)
```

VA 与 VV full-matrix CSV 都保存该 **未做 region 重归一化** 的原始 joint-softmax 概率切片。逐 query 应满足：

```text
sum(VV row) + sum(VA row) ≈ 1
```

### 4.2 VA/VV Sparsity

不能直接在 raw VA 或 raw VV joint mass 上要求累计到 0.9，因为单个 region 的总质量可能小于 0.9。采用与现有 VV 完全兼容的 **region-conditional top-p**：

```text
mass_R(q) = Σ_{k in R} p_joint(k)
p_R(k | R) = p_joint(k) / mass_R(q)
k_R(q) = 达到累计概率 0.9 所需的最少 region key 数
S_R(q) = 1 - k_R(q) / N_R(q)
```

然后对当前 chunk 的 video queries 求均值，得到每个 layer/head/chunk 的 VA 或 VV Sparsity。

为保留“联合 softmax 竞争”信息，`run_va_vv_sparsity.sh` 必须同时导出 region mass：

```text
mean_q mass_VV(q)
mean_q mass_VA(q)
```

每个 head/chunk 上应满足 `mass_VV + mass_VA ≈ 1`。Sparsity 描述 region 内部集中程度，mass 描述 region 在联合 softmax 中获得的总注意力，两者不能混为一个指标。

chunk 0：

- VA key count=`0`。
- VA mass=`0`。
- VA Sparsity 无定义，compact CSV 写字面值 `nan`，并在 `summary.json` 明确记录。

## 5. 输出布局（目标契约）

### 5.1 Sparsity

```text
attn-output/va-vv-sparsity-c10-s24/
└── <prompt-slug>/
    ├── summary.json
    ├── _SUCCESS
    └── step_024/
        ├── sparsity/
        │   ├── vv/l00.csv ... l29.csv
        │   └── va/l00.csv ... l29.csv
        └── mass/
            ├── vv/l00.csv ... l29.csv
            └── va/l00.csv ... l29.csv
```

所有 `lXX.csv`：无表头；rows=heads；columns=chunks。默认形状 `24×10`。

### 5.2 Full attention matrices

```text
attn-output/va-vv-attention-c10-s24/
└── <prompt-slug>/
    ├── summary.json
    ├── _SUCCESS
    └── step_024/
        ├── vv/
        │   ├── l00h00.csv ... l29h23.csv
        └── va/
            ├── l00h00.csv ... l29h23.csv
```

默认共 `2 × 30 × 24 = 1440` 个 matrix CSV：

- VV：每个文件逻辑形状 `2400×2400`。
- VA：每个文件逻辑形状 `2400×288`。
- 行按 chunk 和当前 video Q token 顺序拼接；CSV 第 0 行绘图时位于顶部，行号向下递增。
- 列按对应 region 的 K-history token 首次出现顺序排列。
- 尚不可见的 future keys 写为空字段，不得写 0；0 表示真实的零概率，空字段表示不可用。
- 热力图使用 `origin="upper"`，不得上下翻转矩阵。

## 6. 实现阶段与里程碑

### Phase A — 共享 recorder 核心

- 将当前 `vv_chunk_sparsity` 布尔配置泛化为：
  - `attention_export_mode = matrix | chunk_sparsity`
  - `attention_regions = (VV, VA)`
  - 可选 `attention_heads`
- 抽取一次 streaming joint denominator，并由同一 denominator 物化 VV/VA 两个 raw probability slice。
- 增加 region mass、VA token id、region-specific history metadata。
- 保留现有 VV-only helper 和 CLI 作为兼容层。

**完成标准**：现有 `run_vv_sparsity.sh` 的输出和数值不变；现有 legacy full-matrix payload/writer/reader 回归通过；新的内部 API 可同时返回 VA/VV。

### Phase B — `run_va_vv_sparsity.sh`

- 新增统一 shell 参数解析和默认输出命名。
- GPU 上一次 capture 同时计算 VA/VV conditional top-p Sparsity 与 joint region mass。
- 输出四棵 compact CSV 树，处理 VA chunk 0 的 `nan/0` 约定。

**完成标准**：full CSV 离线复算与 compact 值在 top-p 边界容差内一致。

### Phase C — `run_va_vv_attention.sh`

- 复用 legacy VV block-to-CSV 路径，扩展到 VA。
- 每个 layer/head 分别写 VV 与 VA 原始 joint-softmax CSV。
- 保证两个 region 共享 row 顺序、共享 denominator、各自 K-history 列顺序稳定。
- 增加原子写入、完整 summary 和 `_SUCCESS` 完成标记。

**完成标准**：抽样 query 行满足 `sum(VV)+sum(VA)≈1`；矩阵形状、空字段和方向符合热力图契约。

### Phase D — 文档、验证与交付

- 更新 `attn-scripts/README.md` Script Index、CLI、输出树和数学说明。
- 扩展离线 sparsity 复算工具以按 region 读取新格式，避免把联合列错误混算。
- 本地只做静态检查；提交推送后在 `h100-server` 运行 `uv sync --frozen` 和最小 GPU 数值/端到端验证。

## 7. 预计修改文件

```text
attn-scripts/run_va_vv_sparsity.sh          # 新增
attn-scripts/run_va_vv_attention.sh         # 新增
attn-scripts/README.md                       # 更新索引和用法
wan_va/utils/attention_analysis.py           # 共享 denominator、region metrics、双 region CSV
wan_va/wan_va_server.py                      # 新配置、CLI、校验、完成标记
script/compute_attention_density.py          # region-aware 离线复算
wan_va/utils/attention_sparsity.py           # 如需稳定 tie/共享 helper，则小范围扩展
```

原则：不在 `wan_va/modules/model.py` 堆叠业务逻辑；recorder 仍由现有 hook 接收 query/key/token type/frame id。

## 8. 主要风险

| 风险 | 处理 |
|---|---|
| 把 VA/VV raw mass 直接拿来做 top-p=0.9 | 禁止；Sparsity 必须 region 内重归一化，mass 单独输出 |
| chunk 0 无 VA keys | VA Sparsity=`nan`、mass=`0`、N=`0`，summary 明示 |
| full CSV 体积大 | 仅默认 step 24；支持 layer/head 子集做 smoke；运行前检查至少 60 GiB 可用空间 |
| CSV 中 empty 与 0 混淆 | empty=不可见 future key；0=真实概率零；summary 固化语义 |
| 多 rank 提前出现 summary | 所有 rank 写完并 barrier 后，rank 0 再写 summary 和 `_SUCCESS` |
| 改坏已有 VV 工作流 | 保留兼容参数和回归比较；旧 `run_vv_sparsity.sh` 必须继续通过 |
| 0.9 边界数值差异 | 继续使用 `TOP_P_TOLERANCE=1e-12` 和 fp64 cumulative；允许极少数 ±1 token |

## 9. 详细子计划索引

1. [共享 recorder 与联合 softmax 核心](plans/va_vv_shared_core.md)
2. [VA/VV Sparsity 脚本与 compact 输出](plans/va_vv_sparsity.md)
3. [VA/VV attention matrix 脚本与热力图布局](plans/va_vv_attention.md)
4. [验证、性能与交付流程](plans/va_vv_validation_delivery.md)

## 10. 开始实现前的冻结项

后续实现按本计划默认采用：

- 两个脚本名固定为 `run_va_vv_sparsity.sh` 和 `run_va_vv_attention.sh`。
- Sparsity 为 region-conditional top-p；mass 为 raw joint-softmax region mass。
- full attention CSV 分 `vv/` 与 `va/` 两棵目录，每个 region/layer/head 一个文件。
- query 轴始终是当前 video Q；row 0 在热力图顶部。
- 默认完整任务是 10 chunks、step 24、30 layers、24 heads。

若这些冻结项不变，下一阶段可直接依照子计划实施。
