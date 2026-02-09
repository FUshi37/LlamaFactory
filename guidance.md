# Eyjafjalla（艾雅法拉）角色微调/推理/评测使用指南（LLaMA-Factory）

本指南面向本仓库里与“艾雅法拉（阿黛尔·瑙曼）人格复刻”相关的代码与配置，覆盖：

- 数据集准备（主 SFT + 同人文抽取）
- 注册数据集（`data/dataset_info.json`）
- QLoRA/LoRA 训练（Qwen2.5-7B-Instruct）
- 推理（CLI chat / webchat）
- 回归题库评测（定量追踪人格一致性）

> 约定：以下命令默认在仓库根目录执行：`/home/yangzhe/Project/Eyjafjalla_chat/LlamaFactory`


## 1. 你需要准备什么

- **基础模型**：本地已放置 `models/qwen2.5-7B-Instruct/`
- **训练配置**：`examples/train_lora/eyjafjalla_qwen3_lora_sft.yaml`
- **推理配置**：`examples/inference/eyjafjalla_qwen25_lora_sft.yaml`
- **主数据集**：`data/eyjafjalla_sft.json`（ShareGPT 格式）
- **同人抽取数据集**：`data/eyjafjalla_tongren_sft.json`（ShareGPT 格式，由脚本生成）
- **回归题库**：`data/eyjafjalla_regression.jsonl`
- **回归脚本**：`scripts/run_eyjafjalla_regression.py`
- **同人抽取脚本**：`scripts/build_eyjafjalla_tongren_dataset.py`


## 2. 环境与依赖（最小可运行）

你当前已能运行训练/推理的话，可以跳过本节。

### 2.1 统一使用同一个 Python/CLI（推荐：AICHAT conda 环境）

你这台机器上同时存在系统 Python（例如 `/usr/bin/python3`）与 conda 环境 Python。为了避免出现“脚本缺依赖（例如 `omegaconf`）”的问题，建议本项目所有命令都显式使用你的 conda 环境（你当前使用的是 `AICHAT`）里的可执行文件：

```bash
export LF=/home/yangzhe/miniconda3/envs/AICHAT/bin/llamafactory-cli
export PY=/home/yangzhe/miniconda3/envs/AICHAT/bin/python3
```

后续示例命令里的 `llamafactory-cli` / `python3` 你也可以直接替换为 `$LF` / `$PY` 运行。

### 2.2 依赖检查

建议确认两点：

1) `llamafactory-cli` 可用（在当前环境里）

```bash
cd /home/yangzhe/Project/Eyjafjalla_chat/LlamaFactory
$LF --help
```

2) QLoRA 需要 `bitsandbytes`（webchat 量化推理也会用到）

```bash
$PY -c "import bitsandbytes as bnb; print('bitsandbytes', bnb.__version__)"
```


## 3. 数据格式要点（ShareGPT）

本项目的 SFT 数据用 ShareGPT 格式（`conversations` 列），典型样例如下：

```json
[
  {
    "system": "（可选）系统提示词",
    "conversations": [
      {"from": "human", "value": "用户"},
      {"from": "gpt", "value": "助手"}
    ]
  }
]
```

关键约束（强烈建议遵守）：

- `human` 与 `gpt` **交替出现**（human 在奇数位、gpt 在偶数位）
- SFT 场景下每个样本最好至少包含 **一轮 human + 一轮 gpt**
- 如果你想做“一问多答”，建议做成**多条样本**或通过“追问”做成**多轮对话**，而不是同一轮 human 后跟多个 gpt


## 4. 从同人文抽取训练数据

同人文原始文本位于：

- `Eyjafjalla_data/tongren/*.txt`

运行抽取脚本（默认输入目录与输出路径都已设置好）：

```bash
cd /home/yangzhe/Project/Eyjafjalla_chat/LlamaFactory
$PY scripts/build_eyjafjalla_tongren_dataset.py
```

常用可调参数：

```bash
$PY scripts/build_eyjafjalla_tongren_dataset.py \
  --input_dir Eyjafjalla_data/tongren \
  --output data/eyjafjalla_tongren_sft.json \
  --report saves/tongren_extraction_report.json \
  --max_turns 6 \
  --min_assistant_chars 12 \
  --max_human_chars 360 \
  --max_assistant_chars 420
```

参数含义简述：

- `--max_turns`：每条样本最多多少个 turn（6 = 3 轮问答）
- `--min_assistant_chars`：过滤过短的 assistant 句子，减少噪声
- `--max_human_chars / --max_assistant_chars`：过滤单轮过长的文本块，避免把大段叙述硬塞进对话
- `--allow-nsfw`：允许抽取 NSFW（**不推荐**）
- `--merge-existing`：把新抽取样本去重后合并进既有 ShareGPT JSON（会自动备份 `.bak`）


## 5. 注册数据集（dataset_info.json）

本仓库已将两份数据集注册到 `data/dataset_info.json`：

- `eyjafjalla_sft` -> `data/eyjafjalla_sft.json`
- `eyjafjalla_tongren_sft` -> `data/eyjafjalla_tongren_sft.json`

如果你移动了文件位置，记得同步更新 `file_name`。


## 6. 训练：LoRA / QLoRA（推荐 QLoRA）

训练配置文件：

- `examples/train_lora/eyjafjalla_qwen3_lora_sft.yaml`

你当前配置的要点：

- 基座模型：`/home/yangzhe/Project/Eyjafjalla_chat/LlamaFactory/models/qwen2.5-7B-Instruct`
- 训练阶段：`stage: sft`
- 微调方式：`finetuning_type: lora`
- QLoRA：`quantization_bit: 4`（bnb nf4 + double quant）
- 上下文长度：`cutoff_len: 4096`
- 输出目录：`saves/qwen2.5-7B-Instruct/lora/eyjafjalla_sft_v10`

### 6.1 只用主数据集训练

确保 YAML 里是：

- `dataset: eyjafjalla_sft`

启动训练：

```bash
cd /home/yangzhe/Project/Eyjafjalla_chat/LlamaFactory
CUDA_VISIBLE_DEVICES=0 $LF train examples/train_lora/eyjafjalla_qwen3_lora_sft.yaml
```

### 6.2 主数据集 + 同人数据集一起训（推荐先试）

把训练 YAML 的 `dataset:` 改成：

- `dataset: eyjafjalla_sft,eyjafjalla_tongren_sft`

然后同样启动训练：

```bash
CUDA_VISIBLE_DEVICES=0 $LF train examples/train_lora/eyjafjalla_qwen3_lora_sft.yaml
```

> 提示：LLaMA-Factory 支持多数据集混合；默认混合策略通常是 concat。若你后续想“控制同人占比”，可以再进一步研究 `mix_strategy` / `interleave_probs`（见源码 `src/llamafactory/data/data_utils.py`）。


## 7. 推理：CLI chat 与 webchat

推理配置文件：

- `examples/inference/eyjafjalla_qwen25_lora_sft.yaml`

该配置包含 4-bit 量化参数（bnb nf4），主要目的：

- **避免 webchat 默认的 LoRA 合并导致 OOM**
- 降低显存占用

### 7.1 CLI chat（命令行对话）

```bash
cd /home/yangzhe/Project/Eyjafjalla_chat/LlamaFactory
CUDA_VISIBLE_DEVICES=0 $LF chat examples/inference/eyjafjalla_qwen25_lora_sft.yaml
```

### 7.2 webchat（网页对话）

```bash
cd /home/yangzhe/Project/Eyjafjalla_chat/LlamaFactory
CUDA_VISIBLE_DEVICES=0 $LF webchat examples/inference/eyjafjalla_qwen25_lora_sft.yaml
```

CLI chat vs webchat 的核心差异：

- **CLI chat**：终端交互，轻量，适合快速验证 persona 与解码参数
- **webchat**：网页 UI，更方便演示与长对话；对显存更敏感（所以我们在推理 YAML 加了量化参数）


## 8. 回归评测：用题库定量追踪人格一致性

题库：

- `data/eyjafjalla_regression.jsonl`

评测脚本：

- `scripts/run_eyjafjalla_regression.py`

运行方式（会加载推理 YAML，并逐题生成回答、做 must/must_not/min_chars 等检查）：

```bash
cd /home/yangzhe/Project/Eyjafjalla_chat/LlamaFactory
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_eyjafjalla_regression.py \
  --config examples/inference/eyjafjalla_qwen25_lora_sft.yaml \
  --cases data/eyjafjalla_regression.jsonl \
  --out saves/eyjafjalla_regression_results.jsonl
```

只跑前 N 题用于快速 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_eyjafjalla_regression.py \
  --config examples/inference/eyjafjalla_qwen25_lora_sft.yaml \
  --cases data/eyjafjalla_regression.jsonl \
  --max-cases 10
```

结果文件：

- `saves/eyjafjalla_regression_results.jsonl`

你可以把每次训练的回归结果按时间/版本号另存，长期对比“人格一致性是否在变好”。


## 9. 常见问题（针对你当前项目）

### 9.1 回答冷淡/简短、人物感弱

优先检查三件事：

- 推理 YAML 的 `default_system` 是否足够“角色化”，并且**推理时确实生效**
- 数据集中“日常多轮 + 长答 + 情绪支持”的比例是否足够
- 解码参数是否过于保守（`temperature/top_p/top_k/max_new_tokens`）

### 9.2 输出跑题/自相矛盾

通常用组合拳：

- 数据：补“澄清优先/不确定先追问/用户要求简短就收敛”的样本
- 解码：适当降低随机性（例如降低 `temperature/top_p`）
- 回归题库：新增专门的“纠错后保持一致”测试用例

### 9.3 webchat OOM

确认推理 YAML 中存在：

- `quantization_bit: 4`
- `quantization_method: bnb`
- `quantization_type: nf4`
- `double_quantization: true`

并尽量不要在 webchat 里做 LoRA merge 导出式推理。


## 10. 推荐的日常工作流（最省事）

1) 同人文有新增/脚本更新后，重新抽取：

```bash
$PY scripts/build_eyjafjalla_tongren_dataset.py
```

2) 训练（先混合数据集试跑一版）：

- 修改 `examples/train_lora/eyjafjalla_qwen3_lora_sft.yaml` 的 `dataset:` 为  
  `eyjafjalla_sft,eyjafjalla_tongren_sft`

```bash
CUDA_VISIBLE_DEVICES=0 $LF train examples/train_lora/eyjafjalla_qwen3_lora_sft.yaml
```

3) 推理验证：

```bash
CUDA_VISIBLE_DEVICES=0 $LF chat examples/inference/eyjafjalla_qwen25_lora_sft.yaml
```

4) 回归评测：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_eyjafjalla_regression.py \
  --config examples/inference/eyjafjalla_qwen25_lora_sft.yaml
```

