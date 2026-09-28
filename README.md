# RATE-VQA 项目代码框架

本目录给出论文 **RATE-VQA: Reliability-Aware Temporal Evidence for AIGC Video Quality Assessment** 对应的 PyTorch 工程。代码实现论文中的双时序证据、显式可靠性门控、Qwen3-VL-8B-Instruct 连续 Token 注入、LoRA 训练、五次随机留出、组件消融和 LGVQ→FETV 全量直接迁移协议。

本目录只包含源代码、配置和数据清单模板。数据集、RAFT 权重、Qwen3-VL 权重、缓存特征和训练结果均未放入工程，也不会在创建代码时下载。

## 1. 目录结构

```text
代码/
├─ configs/                    # 论文参数与各数据集/维度配置
├─ schemas/                    # 统一视频清单示例
├─ scripts/
│  ├─ build_manifest.py       # 将原始标注表转换成统一清单
│  ├─ make_splits.py          # 五次视频级 80%/20% 随机划分
│  ├─ prepare_evidence.py     # 拟合训练集阈值并缓存 RAFT 证据
│  ├─ train.py                # 训练单一质量维度模型
│  ├─ evaluate.py             # 数据集内或目标库评价
│  ├─ run_five_splits.py      # 完整五次随机留出流程
│  ├─ run_cross_dataset.py    # 完整 LGVQ→完整 FETV 迁移
│  ├─ run_ablation.py         # 六组结构对齐消融
│  ├─ analyze_reliability.py  # 可靠性分箱与门值统计
│  ├─ aggregate_runs.py       # 五次结果算术平均与范围汇总
│  └─ predict.py              # 单视频推理入口
├─ src/rate_vqa/
│  ├─ data/                   # 视频解码、均匀采样、清单和划分
│  ├─ evidence/               # RAFT、掩膜、双证据和标准化
│  ├─ model/                  # 连续 Token、门控和 Qwen3-VL 集成
│  ├─ checkpoint.py
│  ├─ engine.py
│  ├─ losses.py
│  └─ metrics.py
└─ tests/                     # 不依赖模型权重的公式级测试
```

## 2. 与论文方法的逐项对应

### 2.1 视频采样与时间片

- T2VQA-DB 使用发布的 16 帧、4 FPS 视频。
- LGVQ、GAIA 和 FETV 在完整时域内均匀抽取 `N=min(16,T)` 个真实帧。
- 不做时间插值、循环重复或尾帧补齐。
- 全部 `N-1` 个有效采样相邻帧对按论文公式划入 `K=4` 个连续时间片；16 帧对应的帧对数为 `4/4/4/3`。

### 2.2 RAFT 双向光流与可靠性

`src/rate_vqa/evidence/` 实现：

1. 冻结 RAFT Large，输入帧双线性缩放为 `512×512`；
2. 对全部采样相邻帧计算前向和后向光流；
3. 计算运动归一化前后向循环一致性误差；
4. 使用 `α_occ=0.01` 和 `β_occ=0.5` 生成入界掩膜与高置信掩膜；
5. Flow Evidence 为均值、中位数、90% 分位数和异常比例；
6. Residual Evidence 为帧级残差均值、总体方差和像素残差 90% 分位数；
7. Reliability Descriptor 为入界比例和条件高置信比例。

异常阈值 `τ_abn` 由当前运行训练集全部入界归一化误差的第 90 百分位得到。实现使用磁盘映射数组进行精确分位数计算，阈值在测试和跨库评价阶段保持固定。

### 2.3 连续证据 Token 与门控

- Flow 和 Residual 分别使用独立两层 MLP 投影到语言模型隐藏维度。
- 先进行训练集拟合的 `log1p` 变换和标准化，再加入证据类型嵌入与时间片嵌入。
- 从冻结视觉编码器产生的原生视频 Token 中，按相同时间片端点提取视觉支持并进行注意力池化。
- Flow Gate 与 Residual Gate 参数不共享。
- 完整模型的门输入为：证据 Token、时间对齐视觉上下文、可靠性投影及证据与上下文的逐元素交互。

### 2.4 Qwen3-VL 集成

- 骨干为 `Qwen/Qwen3-VL-8B-Instruct`。
- 原生视觉编码器保持冻结。
- 证据 Token 插入最后一个原生视频 Token 之后、后续文本之前。
- 新 Token 的 attention mask 置为 1；语言主干继续使用原生因果注意力。
- 原生视频 Token 的三轴位置保持不变；证据 Token 使用单调递增且三轴相同的位置，后续文本位置整体顺延。
- 从最后一个有效 EOS 位置读取隐藏状态，并通过线性头输出一个质量预测分数。
- 多维数据库对每个主观维度分别训练一个单输出模型。

### 2.5 参数与训练目标

- LoRA：`r=16`、`alpha=32`、`dropout=0.05`；
- 目标层：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`；
- AdamW：学习率 `1e-5`、weight decay `0.01`；
- 余弦退火，warm-up ratio `0.3`；
- 训练 10 epoch，使用最后一轮模型；
- Smooth-L1 的 `beta=1`；
- 排序损失权重 `0.3`，有效样本对 MOS 差阈值为 5；
- 默认 micro batch 为 2，梯度累积后有效 batch size 为 32；这样每个前向批次均可构造排序样本对。

## 3. 数据清单

每个主观维度准备一份 CSV。必要列如下：

| 列名 | 含义 |
|---|---|
| `video_id` | 数据集内唯一视频编号 |
| `video_path` | 本地视频绝对路径或相对清单的路径 |
| `prompt` | 生成该视频的文本提示词；总体质量没有提示词时可留空 |
| `score` | 当前维度的主观分数 |
| `dataset` | `T2VQA-DB`、`LGVQ`、`GAIA` 或 `FETV` |
| `dimension` | `overall/spatial/temporal/alignment/...` |
| `fps` | 可选；缺失时使用配置中的默认值 |

示例见 `schemas/manifest.example.csv`。原始标注列名不同，可使用：

```powershell
python scripts/build_manifest.py `
  --input D:/dataset/metadata.csv `
  --output data/manifests/lgvq_temporal.csv `
  --video-root D:/dataset/LGVQ `
  --dataset LGVQ `
  --dimension temporal `
  --id-column video_id `
  --path-column file_name `
  --score-column temporal_score `
  --prompt-column prompt
```

同一数据库三个维度的清单应包含相同的 `video_id` 集合。使用同一组 split JSON，即可保证一次运行内各维度模型采用相同划分。

## 4. 权重路径

工程默认在真正执行时通过模型标识加载公开权重。若权重已保存到本地，只需修改配置：

```yaml
model:
  name_or_path: D:/weights/Qwen3-VL-8B-Instruct
  local_files_only: true

evidence:
  raft_local_checkpoint: D:/weights/raft_large.pth
```

设置 `raft_local_checkpoint` 后，代码直接建立未加载预训练权重的 RAFT 结构并读取指定文件，不会再请求 Torchvision 权重。项目目录内不需要存放任何权重。

## 5. 环境入口

以下命令只是后续运行说明，本次代码生成没有执行这些命令。

```powershell
cd "代码"
python -m pip install -e .
```

Qwen3-VL 官方 Transformers 实现要求 `transformers>=4.57.0`。RTX 4090 可使用 BF16 和 SDPA；安装 FlashAttention 后可把 `attn_implementation` 改为 `flash_attention_2`。

## 6. 五次独立随机留出

每次运行重新进行视频级 80%/20% 随机划分，并独立训练模型：

```powershell
python scripts/run_five_splits.py `
  --config configs/t2vqa.yaml `
  --manifest data/manifests/t2vqa_db.csv `
  --work-root outputs/t2vqa
```

每个运行目录包含独立 split、训练集阈值、训练集标准化统计量、最终模型、逐视频预测和指标。`aggregate_metrics.json` 给出五次算术平均、标准差及观测范围。

LGVQ 与 GAIA 的每个维度分别使用对应配置训练。为保证多维模型共享划分，可先建立一次 split，再对所有维度复用同一组 split 文件。

## 7. 分阶段运行

```powershell
python scripts/make_splits.py `
  --config configs/t2vqa.yaml `
  --manifest data/manifests/t2vqa_db.csv `
  --output outputs/t2vqa/splits

python scripts/prepare_evidence.py `
  --config configs/t2vqa.yaml `
  --train-manifest data/manifests/t2vqa_db.csv `
  --transform-manifest data/manifests/t2vqa_db.csv `
  --split outputs/t2vqa/splits/run_01.json `
  --output outputs/t2vqa/run_01/evidence

python scripts/train.py `
  --config configs/t2vqa.yaml `
  --manifest data/manifests/t2vqa_db.csv `
  --split outputs/t2vqa/splits/run_01.json `
  --evidence outputs/t2vqa/run_01/evidence `
  --output outputs/t2vqa/run_01/training

python scripts/evaluate.py `
  --config configs/t2vqa.yaml `
  --manifest data/manifests/t2vqa_db.csv `
  --split outputs/t2vqa/splits/run_01.json `
  --subset test `
  --evidence outputs/t2vqa/run_01/evidence `
  --checkpoint outputs/t2vqa/run_01/training/final/rate_vqa.pt `
  --output outputs/t2vqa/run_01/evaluation
```

T2VQA-DB 在计算 PLCC 和 RMSE 前执行四参数逻辑映射，SRCC 和 KRCC 使用原始预测顺序。LGVQ、GAIA 和跨库配置默认不执行该映射。

## 8. LGVQ→FETV 全量直接迁移

每个维度单独运行。源模型使用完整 LGVQ 训练，并在完整 FETV 上直接评价。FETV 标签仅用于最终指标计算：

```powershell
python scripts/run_cross_dataset.py `
  --config configs/lgvq_temporal.yaml `
  --source-manifest data/manifests/lgvq_temporal.csv `
  --target-manifest data/manifests/fetv_temporal.csv `
  --work-root outputs/cross_lgvq_fetv/temporal
```

空间、时序、对齐三个迁移模型分别运行。源训练得到的 `τ_abn`、证据变换、标准化统计量和模型参数均直接用于 FETV。

## 9. 消融实验

`run_ablation.py` 固定五组划分、输入采样、训练目标和预算，仅改变证据注入及门控输入：

| Case | `evidence_mode` | `gate_mode` | 含义 |
|---|---|---|---|
| (a) | `none` | `none` | Visual baseline |
| (b) | `flow` | `none` | Flow Evidence |
| (c) | `residual` | `none` | Residual Evidence |
| (d) | `dual` | `none` | 双证据直接注入 |
| (e) | `dual` | `context` | Context-conditioned Gate |
| (f) | `dual` | `reliability` | Context + Explicit Reliability |

运行示例：

```powershell
python scripts/run_ablation.py `
  --config configs/t2vqa.yaml `
  --manifest data/manifests/t2vqa_db.csv `
  --split-root outputs/t2vqa/splits `
  --evidence-root outputs/t2vqa `
  --output-root outputs/t2vqa_ablation
```

## 10. 可靠性机制分析

完整模型与 Context-only 模型评价后，可按 `r_conf` 等频分成五组，汇总两类门值、95% 置信区间与 MAE：

```powershell
python scripts/analyze_reliability.py `
  --manifest data/manifests/t2vqa_db.csv `
  --full-predictions outputs/full/evaluation/predictions.csv `
  --context-predictions outputs/context/evaluation/predictions.csv `
  --evidence outputs/full/evidence `
  --output outputs/reliability_analysis
```

脚本输出逐时间片数据和分箱汇总表，可直接用于论文中的可靠性感知机制图。

## 11. 单视频推理

单视频推理必须复用某个已训练源模型对应的训练集阈值和标准化统计量：

```powershell
python scripts/predict.py `
  --config configs/t2vqa.yaml `
  --checkpoint outputs/t2vqa/run_01/training/final/rate_vqa.pt `
  --evidence-stats outputs/t2vqa/run_01/evidence `
  --video D:/videos/example.mp4 `
  --prompt "A skier moves down a snowy slope."
```

## 12. 实现边界

- 代码不把可靠性比例直接解释为质量分数，而是将其作为门控条件。
- Residual-only 配置仍调用 RAFT 完成变形和置信掩膜构造，只是不注入 Flow Token。
- 每次数据集内运行单独拟合训练集阈值和标准化统计量，测试集不参与这些步骤。
- 跨库流程只在完整 LGVQ 上拟合阈值、标准化和模型参数，并原样用于完整 FETV。
- 代码不会替代原始数据库的授权、下载与目录组织，清单中的路径始终指向用户已有的本地文件。

## 13. 上游接口依据

- Qwen3-VL 官方仓库：<https://github.com/QwenLM/Qwen3-VL>
- Hugging Face Qwen3-VL 模型接口：<https://huggingface.co/docs/transformers/model_doc/qwen3_vl>
- Torchvision RAFT 接口：<https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html>
