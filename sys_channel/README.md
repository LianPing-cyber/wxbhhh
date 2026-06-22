# Audit TF-IDF + PCAP BGE-M3 融合分类（推理包）

只需验证/推理，不训练。模型由维护者在训练集上导出后，与本目录一并交付。

**当前模型版本**：`audit_tfidf_pcap_bge_m3_fusion_v2`（2360 条双模态样本训练）

## 目录结构

```text
audit_tfidf_pcap_fusion/
  validate.py              # 运行：加载模型 + 在新数据上验证
  train_export_model.py    # 维护者运行：训练并导出 model
  requirements.txt
  README.md
  sync_lib.sh              # 从主仓库 scripts/ 同步 lib/
  model/                   # 已训练参数（必含）
    model_config.json
    svm.joblib
    feature_artifacts.joblib
  lib/                     # 特征提取与推理逻辑
    audit_pcap_fusion_svm_classifier.py
    tokenize_audit_log.py
```

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

PCAP 语义向量使用 BGE-M3，首次运行会从 ModelScope 下载到：

`~/.cache/modelscope/hub/models/BAAI/bge-m3`

也可指定本地模型：

```bash
export BGE_MODEL_PATH=/path/to/bge-m3
```

## 2. 验证数据格式

`--data-root` 指向**新数据集**根目录：

```text
<data-root>/
  innocent-by-type/
    audit-logs/*.log
    network/*_network.pcap
    audit_token_per_file.csv    # 可选，有则更快
  malicious-by-type/
    （同上）
```

文件名约定：`{skill}_prompt_{N}_audit-logs.log` 与 `{skill}_prompt_{N}_network.pcap` 对齐。

**v2 模型要求**：每条样本必须同时有 audit 日志与 pcap 文件，缺任一模态的 session 会被自动跳过（与训练集筛选规则一致）。

## 3. 一键验证

```bash
python validate.py --data-root /path/to/your_validation_data
```

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--data-root` | 验证集根目录 | **必填** |
| `--model-dir` | 模型目录 | `./model` |
| `--output-dir` | 结果输出 | `<data-root>/comparison/validation_results` |
| `--cache-dir` | PCAP BGE 缓存 | `<data-root>/comparison` |
| `--require-both-modalities` / `--no-require-both-modalities` | 是否仅保留双模态样本 | 跟随 `model_config.json`（v2 为 True） |

### 示例

```bash
python validate.py \
  --data-root /data/project_b/val \
  --model-dir ./model \
  --output-dir /data/project_b/val/results
```

## 4. 输出文件

- `predictions.csv` — 每条样本的预测标签、是否正确（有真值时）
- `validation_metrics.json` — accuracy / precision / recall / f1
- `confusion_matrix.csv` — 混淆矩阵

## 5. 维护者：重新导出模型（换训练集时）

在**训练集**上执行一次（默认仅保留 audit∩pcap 双模态样本）：

```bash
python train_export_model.py --data-root /path/to/train_data --model-dir ./model
```


同步主仓库脚本到 lib/：

```bash
./sync_lib.sh
```

## 6. 模型说明

| 组件 | 内容 |
|------|------|
| `svm.joblib` | 训练好的 LinearSVC |
| `feature_artifacts.joblib` | Audit TF-IDF 向量器 + 数值 MinMaxScaler |
| `model_config.json` | 编码参数、BGE 配置、特征名、样本筛选规则等 |
| 运行时 BGE | PCAP 文本在推理时用 BGE-M3 现场编码（参数见 config） |

| 配置项 | v2 值 |
|--------|-------|
| Audit 编码 | TF-IDF |
| PCAP 编码 | BGE-M3 |
| 融合方式 | 早期融合（fusion） |
| 样本筛选 | `require_both_modalities=true`（audit∩pcap 交集） |
| 训练样本数 | 2360（innocent=1216, malicious=1144） |

## 7. 版本变更（v1 → v2）

| 项目 | v1 | v2 |
|------|----|----|
| 样本范围 | audit∪pcap 并集（2553 条） | audit∩pcap 交集（2360 条） |
| 单模态 session | 保留，缺失模态特征置空 | 训练/验证时排除 |
| bundle 版本 | `audit_tfidf_pcap_bge_m3_fusion_v1` | `audit_tfidf_pcap_bge_m3_fusion_v2` |

上线时请整包替换（含 `model/` 三个文件），不要只换 `svm.joblib`。

使用时解压后：

```bash
cd audit_tfidf_pcap_fusion
pip install -r requirements.txt
python validate.py --data-root /path/to/validation_data
```

首次推理会自动下载 BGE-M3（约 2GB），建议提前配置 `BGE_MODEL_PATH` 或保证可访问 ModelScope。
