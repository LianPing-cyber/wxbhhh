#!/usr/bin/env python3
"""
Auditd + PCAP 早期融合 + 线性 SVM 二分类（innocent vs malicious）。

每个 Skill Session 天然对齐一对文件：
  - {session}_audit-logs.log
  - {session}_network.pcap

特征层:
  1. 主机语义（audit 词元）：TF-IDF 或 BGE-M3 稠密向量（见 AUDIT_ENCODER）
  2. 主机数值（失败 syscall 比例、execve 失败率等）
  3. 网络语义（DNS / TLS SNI / HTTP Host 域名）：TF-IDF 或 BGE-M3 稠密向量（见 NET_ENCODER）
  4. 网络数值（包数、字节、上下行比、包长/IAT 统计）

Audit BGE-M3 优先复用 comparison/audit_bge_m3_embeddings.npz（mean-pool，与 audit_svm_classifier.py 同源）；
max-pool 时使用 comparison/audit_bge_m3_max_embeddings.npz（首次编码后写入，含 fingerprint）。
PCAP BGE-M3 缓存于 comparison/pcap_bge_m3_embeddings.npz（首次运行编码后写入）。
长文本分块编码（chunk=384 token）避免 OOM，仅对缓存中缺失的 audit 样本按需补编码。

融合策略（方案 A）:
  稀疏 TF-IDF 与稠密数值向量横向拼接；数值部分 log1p + MinMaxScaler(0,1)（每折仅在训练集 fit），
  缩放后再按 HOST/NET 权重调节相对 TF-IDF 的贡献。

在 IDE 中直接运行本文件；修改下方 RUN_MODE 切换 ablation：
  - all        : 统一样本（audit∩pcap）后依次跑 fusion → audit_only → pcap_only
  - fusion     : 默认 audit∩pcap 交集（REQUIRE_BOTH_MODALITIES=True）；关闭后可用 audit∪pcap 并集
  - audit_only : 仅读 audit 日志/CSV
  - pcap_only  : 仅读 network/*.pcap

十折交叉验证使用 StratifiedGroupKFold，按 skill 分组（同一 skill 的多条 prompt
不会同时出现在训练集与测试集）。

输出:
  - comparison/audit_tfidf_pcap_encoder/fusion_svm_{mode}_*.csv
  - comparison/audit_tfidf_pcap_encoder/fusion_svm_*_sample_manifest.csv
  - BGE 编码缓存仍复用 comparison/*.npz（不写入结果目录）
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import struct
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from sys_channel.lib.tokenize_audit_log import (  # noqa: E402
    count_tokens_in_audit_file,
    should_skip_token,
)

# ===========================================================================
# 运行配置
# ===========================================================================

INNOCENT_AUDIT_DIR = PROJECT_ROOT / "innocent-by-type" / "audit-logs"
MALICIOUS_AUDIT_DIR = PROJECT_ROOT / "malicious-by-type" / "audit-logs"
INNOCENT_NETWORK_DIR = PROJECT_ROOT / "innocent-by-type" / "network"
MALICIOUS_NETWORK_DIR = PROJECT_ROOT / "malicious-by-type" / "network"
INNOCENT_TOKEN_CSV = PROJECT_ROOT / "innocent-by-type" / "audit_token_per_file.csv"
MALICIOUS_TOKEN_CSV = PROJECT_ROOT / "malicious-by-type" / "audit_token_per_file.csv"
COMPARISON_DIR = PROJECT_ROOT / "comparison"
OUTPUT_DIR = COMPARISON_DIR / "audit_tfidf_pcap_encoder"

EvalMode = Literal["fusion", "audit_only", "pcap_only"]
RunMode = Literal["fusion", "audit_only", "pcap_only", "all"]
RUN_MODE: RunMode = "fusion"

# 单模态实验时是否与 all 一样使用 audit∩pcap 交集
USE_UNIFIED_SAMPLES = False
# fusion 模式是否仅保留 audit 与 pcap 均存在的样本（上线包默认 True）
REQUIRE_BOTH_MODALITIES = False

UNIFIED_MANIFEST_PATH = OUTPUT_DIR / "fusion_svm_unified_sample_manifest.csv"
FUSION_MANIFEST_PATH = OUTPUT_DIR / "fusion_svm_fusion_sample_manifest.csv"

PREFER_TOKEN_CSV = True
MAX_TOKEN_REPEAT = 30

# 语义编码: "tfidf" | "bge-m3"
SemanticEncoderName = Literal["tfidf", "bge-m3"]
AUDIT_ENCODER: SemanticEncoderName = "tfidf"
NET_ENCODER: SemanticEncoderName = "bge-m3"

# TF-IDF
AUDIT_TFIDF_MAX_FEATURES = 100_000
NET_TFIDF_MAX_FEATURES = 5_000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# BGE-M3（与 audit_svm_classifier.py 对齐；缓存由该脚本生成）
BgeChunkPooling = Literal["mean", "max"]


def resolve_bge_model_path() -> str:
    """优先读环境变量 BGE_MODEL_PATH / BGE_MODEL，否则用 ModelScope 默认缓存路径。"""
    for key in ("BGE_MODEL_PATH", "BGE_MODEL"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return str(Path.home() / ".cache/modelscope/hub/models/BAAI/bge-m3")


BGE_MODEL = resolve_bge_model_path()
BGE_DEVICE = "cpu"  # 长 audit 建议 cpu，避免 MPS/CUDA OOM
BGE_CHUNK_TOKENS = 384  # 每块最大 BPE token 数（model.max_seq_length）
BGE_CHUNK_STRIDE = 384
BGE_BATCH_SIZE = 8  # 按「块」批处理
BGE_NORMALIZE = True
BGE_MAX_CHUNKS = 0
# mean: 与 audit_svm 缓存兼容；max: 保留局部异常块信号（需重编码 audit 缓存）
BGE_CHUNK_POOLING: BgeChunkPooling = "mean"
BGE_ENCODING_VERSION = 3
BGE_CACHE_PATH = COMPARISON_DIR / "audit_bge_m3_embeddings.npz"  # mean-pool（audit_svm 同源）
AUDIT_BGE_MAX_CACHE_PATH = COMPARISON_DIR / "audit_bge_m3_max_embeddings.npz"
AUDIT_BGE_USE_CACHE = True
BGE_AUDIT_MAX_TOKEN_REPEAT = 3  # 须与 audit_svm 缓存 fingerprint 中 repeat 一致
PCAP_BGE_CACHE_PATH = COMPARISON_DIR / "pcap_bge_m3_embeddings.npz"
PCAP_BGE_ENCODING_VERSION = 2
PCAP_BGE_USE_CACHE = True

# 数值特征权重：在 MinMaxScaler 之后施加，方可真正调节相对 TF-IDF 的贡献
# （硬编码权重是 baseline 取舍；自适应调参留作后续 GridSearch / MKL 扩展）
HOST_NUMERIC_WEIGHT = 1.0
NET_NUMERIC_WEIGHT = 2.0

# PCAP 特征提取版本（变更解析逻辑时可递增，便于日志排查）
PCAP_FEATURE_VERSION = 3
MAX_TCP_STREAM_BUF = 65_536
MAX_TCP_STREAMS = 2_048  # 单 pcap 最多跟踪的 TCP 流数（LRU 淘汰）

# SVM
SVM_C = 1.0
SVM_CLASS_WEIGHT = "balanced"

N_FOLDS = 10
RANDOM_STATE = 42

# 0 = 全量；>0 时每类最多取 N 条（调试）
MAX_SAMPLES_PER_CLASS = 0

LABEL_INNOCENT = 0
LABEL_MALICIOUS = 1
LABEL_NAMES = {LABEL_INNOCENT: "innocent", LABEL_MALICIOUS: "malicious"}


def extract_audit_skill_id(file_name: str) -> str:
    """从 audit 日志文件名解析 skill 标识（同一 skill 的 prompt_0/1/2/3 共享）。"""
    match = re.match(r"^(.+)_prompt_\d+_audit-logs\.log$", file_name)
    if match:
        return match.group(1)
    return Path(file_name).stem


def build_cv_groups(file_names: list[str], labels: list[int]) -> np.ndarray:
    """按 (标签, skill) 分组，避免 innocent/malicious 同名 skill 混为一组。"""
    groups = [
        f"{LABEL_NAMES[label]}:{extract_audit_skill_id(name)}"
        for name, label in zip(file_names, labels, strict=True)
    ]
    return np.array(groups, dtype=object)


def build_cv_groups_from_samples(samples: list[SampleFeatures]) -> np.ndarray:
    return build_cv_groups(
        [sample.file_name for sample in samples],
        [sample.label for sample in samples],
    )


def validate_group_cv(groups: np.ndarray, y: np.ndarray, n_splits: int) -> None:
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise SystemExit(
            f"CV 分组数 ({len(unique_groups)}) 少于折数 ({n_splits})，"
            f"无法执行 {n_splits} 折 StratifiedGroupKFold"
        )

    group_label_sets: dict[str, set[int]] = defaultdict(set)
    for group, label in zip(groups, y, strict=True):
        group_label_sets[str(group)].add(int(label))
    mixed = [group for group, label_set in group_label_sets.items() if len(label_set) > 1]
    if mixed:
        raise SystemExit(
            "CV 分组含混合标签（实现错误）: "
            + ", ".join(mixed[:5])
        )

    group_sizes: dict[str, int] = defaultdict(int)
    for group in groups:
        group_sizes[str(group)] += 1
    sizes = list(group_sizes.values())
    log_progress(
        f"CV 分组: {len(unique_groups)} 个 skill 组, "
        f"组内样本 min/median/max="
        f"{min(sizes)}/{int(np.median(sizes))}/{max(sizes)}"
    )


def assert_no_group_leakage(
    groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    train_groups = set(groups[train_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    leaked = train_groups & test_groups
    if leaked:
        sample = sorted(leaked)[:5]
        raise RuntimeError(f"CV 分组泄漏: {sample}")


def make_group_stratified_kfold() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )


# audit log 解析
RE_EXECVE_LINE = re.compile(r"type=EXECVE\b")
RE_EXECVE_HEX_ARG = re.compile(r"\ba(\d+)=([0-9A-Fa-f]{8,})\b")
DOMAIN_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$"
)

HOST_NUMERIC_NAMES = [
    "syscall_total",
    "syscall_failed",
    "syscall_exit_neg2",
    "syscall_failed_ratio",
    "execve_total",
    "execve_failed",
    "execve_failed_ratio",
    "execve_payload_chars",
]

NET_NUMERIC_NAMES = [
    "packet_count",
    "total_bytes",
    "up_bytes",
    "down_bytes",
    "up_ratio",
    "pkt_len_min",
    "pkt_len_max",
    "pkt_len_mean",
    "pkt_len_std",
    "iat_mean",
    "iat_std",
    "iat_max",
    "dns_domain_count",
    "sni_count",
    "http_host_count",
]

# fusion 专用：不参与 MinMaxScaler，显式区分「模态缺失」与「数值为零」
MODALITY_FLAG_NAMES = ["audit_present", "pcap_present"]


@dataclass
class SampleFeatures:
    file_name: str
    label: int
    audit_text: str
    host_numeric: np.ndarray
    net_text: str
    net_numeric: np.ndarray
    has_audit: bool
    has_pcap: bool


@dataclass
class FusionFeatureArtifacts:
    """训练阶段 fit 的变换器（TF-IDF / MinMaxScaler）。"""

    mode: EvalMode
    audit_encoder: SemanticEncoderName
    net_encoder: SemanticEncoderName
    audit_tfidf: TfidfVectorizer | None = None
    net_tfidf: TfidfVectorizer | None = None
    numeric_scaler: MinMaxScaler | None = None


MODEL_BUNDLE_VERSION = "audit_tfidf_pcap_bge_m3_fusion_v2"
DEFAULT_MODEL_DIR_NAME = "model_bundle"


def log_progress(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def validate_run_mode(mode: str) -> RunMode:
    if mode not in ("fusion", "audit_only", "pcap_only", "all"):
        raise SystemExit(
            f'RUN_MODE 无效: {mode!r}，请设为 "fusion" | "audit_only" | "pcap_only" | "all"'
        )
    return mode  # type: ignore[return-value]


def validate_semantic_encoder(name: str, param: str) -> SemanticEncoderName:
    if name not in ("tfidf", "bge-m3"):
        raise SystemExit(f'{param} 无效: {name!r}，请设为 "tfidf" 或 "bge-m3"')
    return name  # type: ignore[return-value]


def validate_audit_encoder(name: str) -> SemanticEncoderName:
    return validate_semantic_encoder(name, "AUDIT_ENCODER")


def validate_net_encoder(name: str) -> SemanticEncoderName:
    return validate_semantic_encoder(name, "NET_ENCODER")


def output_paths(mode: EvalMode) -> tuple[Path, Path, Path, Path]:
    return (
        OUTPUT_DIR / f"fusion_svm_{mode}_cv_results.csv",
        OUTPUT_DIR / f"fusion_svm_{mode}_cv_summary.csv",
        OUTPUT_DIR / f"fusion_svm_{mode}_confusion_matrix.csv",
        OUTPUT_DIR / f"fusion_svm_{mode}_misclassified.csv",
    )


def audit_name_to_pcap_name(audit_name: str) -> str:
    if audit_name.endswith("_audit-logs.log"):
        return audit_name.replace("_audit-logs.log", "_network.pcap")
    return audit_name.removesuffix(".log") + "_network.pcap"


def pcap_name_to_audit_name(pcap_name: str) -> str:
    if pcap_name.endswith("_network.pcap"):
        return pcap_name.replace("_network.pcap", "_audit-logs.log")
    return pcap_name.removesuffix(".pcap") + "_audit-logs.log"


def list_pcap_session_names(network_dir: Path) -> list[str]:
    """从 network 目录枚举 pcap，返回与之对齐的 audit 风格文件名（仅作样本 ID）。"""
    if not network_dir.is_dir():
        return []
    return [
        pcap_name_to_audit_name(path.name)
        for path in sorted(network_dir.glob("*_network.pcap"))
    ]


def list_audit_session_names(audit_dir: Path) -> list[str]:
    if not audit_dir.is_dir():
        return []
    return sorted(path.name for path in audit_dir.glob("*.log"))


def unify_session_names(
    audit_dir: Path,
    network_dir: Path,
    class_name: str,
) -> list[str]:
    """取 audit-logs 与 network/*.pcap 均存在的 Session（交集）。"""
    audit_names = set(list_audit_session_names(audit_dir))
    pcap_names = set(list_pcap_session_names(network_dir))
    unified = sorted(audit_names & pcap_names)
    only_audit = len(audit_names - pcap_names)
    only_pcap = len(pcap_names - audit_names)
    log_progress(
        f"{class_name} 统一样本: {len(unified)} 个 "
        f"(audit={len(audit_names)}, pcap={len(pcap_names)}, "
        f"仅audit={only_audit}, 仅pcap={only_pcap})"
    )
    return unified


def union_session_names(
    audit_dir: Path,
    network_dir: Path,
    class_name: str,
) -> list[str]:
    """取 audit-logs 与 network/*.pcap 的并集（fusion 全量）。"""
    audit_names = set(list_audit_session_names(audit_dir))
    pcap_names = set(list_pcap_session_names(network_dir))
    union = sorted(audit_names | pcap_names)
    both = len(audit_names & pcap_names)
    only_audit = len(audit_names - pcap_names)
    only_pcap = len(pcap_names - audit_names)
    log_progress(
        f"{class_name} fusion 全量: {len(union)} 个 "
        f"(audit={len(audit_names)}, pcap={len(pcap_names)}, "
        f"双模态={both}, 仅audit={only_audit}, 仅pcap={only_pcap})"
    )
    return union


def write_fusion_manifest(
    inn_names: list[str],
    mal_names: list[str],
    inn_audit_dir: Path,
    mal_audit_dir: Path,
    inn_network_dir: Path,
    mal_network_dir: Path,
) -> None:
    rows: list[dict[str, str | bool]] = []
    for name in inn_names:
        rows.append(_fusion_manifest_row(
            name, LABEL_NAMES[LABEL_INNOCENT],
            inn_audit_dir, inn_network_dir,
        ))
    for name in mal_names:
        rows.append(_fusion_manifest_row(
            name, LABEL_NAMES[LABEL_MALICIOUS],
            mal_audit_dir, mal_network_dir,
        ))
    write_csv(
        FUSION_MANIFEST_PATH,
        rows,
        ["file_name", "label", "has_audit", "has_pcap", "pcap_file"],
    )
    log_progress(f"fusion 样本清单: {FUSION_MANIFEST_PATH.name} ({len(rows)} 条)")


def _fusion_manifest_row(
    file_name: str,
    label: str,
    audit_dir: Path,
    network_dir: Path,
) -> dict[str, str | bool]:
    has_audit = (audit_dir / file_name).is_file()
    pcap_file = audit_name_to_pcap_name(file_name)
    has_pcap = (network_dir / pcap_file).is_file()
    return {
        "file_name": file_name,
        "label": label,
        "has_audit": has_audit,
        "has_pcap": has_pcap,
        "pcap_file": pcap_file,
    }


def write_unified_manifest(inn_names: list[str], mal_names: list[str]) -> None:
    rows: list[dict[str, str]] = []
    for name in inn_names:
        rows.append(
            {
                "file_name": name,
                "label": LABEL_NAMES[LABEL_INNOCENT],
                "pcap_file": audit_name_to_pcap_name(name),
            }
        )
    for name in mal_names:
        rows.append(
            {
                "file_name": name,
                "label": LABEL_NAMES[LABEL_MALICIOUS],
                "pcap_file": audit_name_to_pcap_name(name),
            }
        )
    write_csv(
        UNIFIED_MANIFEST_PATH,
        rows,
        ["file_name", "label", "pcap_file"],
    )
    log_progress(f"统一样本清单: {UNIFIED_MANIFEST_PATH.name} ({len(rows)} 条)")


def resolve_sample_names(
    run_mode: RunMode,
    use_unified: bool,
    require_both: bool = False,
) -> tuple[list[str], list[str], bool]:
    """返回 (inn_names, mal_names, unified_flag)。"""
    if use_unified or run_mode == "all" or (run_mode == "fusion" and require_both):
        inn_names = unify_session_names(
            INNOCENT_AUDIT_DIR, INNOCENT_NETWORK_DIR, "innocent"
        )
        mal_names = unify_session_names(
            MALICIOUS_AUDIT_DIR, MALICIOUS_NETWORK_DIR, "malicious"
        )
        return inn_names, mal_names, True
    if run_mode == "fusion":
        inn_names = union_session_names(
            INNOCENT_AUDIT_DIR, INNOCENT_NETWORK_DIR, "innocent"
        )
        mal_names = union_session_names(
            MALICIOUS_AUDIT_DIR, MALICIOUS_NETWORK_DIR, "malicious"
        )
        return inn_names, mal_names, False
    if run_mode == "pcap_only":
        log_progress("pcap_only: 按 network/*.pcap 枚举样本")
        return (
            list_pcap_session_names(INNOCENT_NETWORK_DIR),
            list_pcap_session_names(MALICIOUS_NETWORK_DIR),
            False,
        )
    log_progress("按 audit-logs/*.log 枚举样本")
    return (
        list_audit_session_names(INNOCENT_AUDIT_DIR),
        list_audit_session_names(MALICIOUS_AUDIT_DIR),
        False,
    )


def counter_to_document(
    token_counts: list[tuple[str, int]],
    max_repeat: int | None = None,
) -> str:
    repeat_cap = MAX_TOKEN_REPEAT if max_repeat is None else max_repeat
    parts: list[str] = []
    for token, count in token_counts:
        if should_skip_token(token):
            continue
        repeat = min(max(count, 1), repeat_cap)
        parts.extend([token] * repeat)
    return " ".join(parts)


def split_text_into_bge_chunks(
    text: str,
    tokenizer,
    max_tokens: int,
    stride: int,
    max_chunks: int = 0,
) -> list[str]:
    """按 BGE tokenizer 将长文档切分为多块（短于 max_tokens 则整块保留）。"""
    if not text.strip():
        return [""]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks: list[str] = []
    for start in range(0, len(token_ids), stride):
        piece = token_ids[start : start + max_tokens]
        if not piece:
            break
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))
        if start + max_tokens >= len(token_ids):
            break
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break
    return chunks or [""]


def resolve_bge_device(request: str) -> str:
    import torch

    if request == "cpu":
        return "cpu"
    if request == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        log_progress("MPS 不可用，回退 CPU")
        return "cpu"
    if request == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    log_progress(f"未知 BGE_DEVICE={request!r}，使用 CPU")
    return "cpu"


def _encode_bge_chunks(model, chunks: list[str], device: str) -> np.ndarray:
    """编码所有块；OOM 时自动回退 CPU。"""
    import torch

    def _run(active_model) -> np.ndarray:
        return active_model.encode(
            chunks,
            batch_size=BGE_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )

    try:
        return np.asarray(_run(model), dtype=np.float64)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" not in msg and "invalid buffer size" not in msg:
            raise
        if device == "cpu":
            raise
        log_progress(f"设备 {device} 内存不足 ({exc!s})，回退 CPU 重新编码 …")
        if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        from sentence_transformers import SentenceTransformer

        cpu_model = SentenceTransformer(
            BGE_MODEL, trust_remote_code=True, device="cpu"
        )
        cpu_model.max_seq_length = BGE_CHUNK_TOKENS
        return np.asarray(_run(cpu_model), dtype=np.float64)


def pool_chunk_embeddings(
    chunk_embeddings: np.ndarray,
    chunk_owner: list[int],
    n_docs: int,
    pooling: BgeChunkPooling,
) -> np.ndarray:
    """将分块向量聚合为文档级向量。"""
    dim = chunk_embeddings.shape[1]
    if pooling == "mean":
        pooled = np.zeros((n_docs, dim), dtype=np.float64)
        counts = np.zeros(n_docs, dtype=np.float64)
        for emb, doc_idx in zip(chunk_embeddings, chunk_owner, strict=True):
            pooled[doc_idx] += emb
            counts[doc_idx] += 1.0
        counts = np.maximum(counts, 1.0)
        pooled /= counts[:, np.newaxis]
        return pooled
    if pooling == "max":
        pooled = np.full((n_docs, dim), -np.inf, dtype=np.float64)
        for emb, doc_idx in zip(chunk_embeddings, chunk_owner, strict=True):
            pooled[doc_idx] = np.maximum(pooled[doc_idx], emb)
        empty_docs = ~np.isfinite(pooled).any(axis=1)
        pooled[empty_docs] = 0.0
        return pooled
    raise SystemExit(f"BGE_CHUNK_POOLING 无效: {pooling!r}")


def encode_with_bge_m3(texts: list[str]) -> np.ndarray:
    """用冻结 BGE-M3 分块编码文档，按 BGE_CHUNK_POOLING 聚合块向量。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "BGE-M3 需要 sentence-transformers 与 torch，请安装:\n"
            "  pip install sentence-transformers torch"
        ) from exc

    device = resolve_bge_device(BGE_DEVICE)
    log_progress(f"加载 BGE-M3 模型: {BGE_MODEL} (device={device})")
    model = SentenceTransformer(BGE_MODEL, trust_remote_code=True, device=device)
    model.max_seq_length = BGE_CHUNK_TOKENS
    tokenizer = model.tokenizer

    doc_chunks: list[list[str]] = [
        split_text_into_bge_chunks(
            text, tokenizer, BGE_CHUNK_TOKENS, BGE_CHUNK_STRIDE, BGE_MAX_CHUNKS
        )
        for text in texts
    ]
    flat_chunks: list[str] = []
    chunk_owner: list[int] = []
    chunk_counts = [len(chunks) for chunks in doc_chunks]
    for doc_idx, chunks in enumerate(doc_chunks):
        for chunk in chunks:
            flat_chunks.append(chunk)
            chunk_owner.append(doc_idx)

    total_chunks = len(flat_chunks)
    multi_chunk_docs = sum(1 for n in chunk_counts if n > 1)
    max_chunks = max(chunk_counts) if chunk_counts else 0
    log_progress(
        f"BGE-M3 分块: {len(texts)} 文档 → {total_chunks} 块 "
        f"(multi_chunk_docs={multi_chunk_docs}, max_chunks_per_doc={max_chunks}, "
        f"chunk_tokens={BGE_CHUNK_TOKENS}, batch={BGE_BATCH_SIZE}, "
        f"pool={BGE_CHUNK_POOLING})"
    )

    t0 = time.perf_counter()
    chunk_embeddings = _encode_bge_chunks(model, flat_chunks, device)
    elapsed_encode = time.perf_counter() - t0

    pooled = pool_chunk_embeddings(
        chunk_embeddings, chunk_owner, len(texts), BGE_CHUNK_POOLING
    )

    if BGE_NORMALIZE:
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-12)

    log_progress(
        f"BGE-M3 编码完成: shape={pooled.shape}, "
        f"耗时 {elapsed_encode:.1f}s ({total_chunks} 块)"
    )
    return pooled


def load_bge_cache_map(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.is_file():
        raise SystemExit(f"BGE 缓存不存在: {cache_path}")
    cached = np.load(cache_path, allow_pickle=True)
    names = [str(n) for n in cached["file_names"]]
    embeddings = cached["embeddings"]
    log_progress(
        f"加载 BGE 缓存: {cache_path.name} "
        f"({len(names)} 条, dim={embeddings.shape[1]})"
    )
    return dict(zip(names, embeddings, strict=True))


def pcap_encoding_config_tag() -> str:
    return (
        f"pcap_v{PCAP_BGE_ENCODING_VERSION}|pcap_feat={PCAP_FEATURE_VERSION}|"
        f"bge_v{BGE_ENCODING_VERSION}|pool={BGE_CHUNK_POOLING}|"
        f"chunk={BGE_CHUNK_TOKENS}|stride={BGE_CHUNK_STRIDE}|"
        f"batch={BGE_BATCH_SIZE}|dev={BGE_DEVICE}|norm={BGE_NORMALIZE}|"
        f"max_chunks={BGE_MAX_CHUNKS}"
    )


def pcap_corpus_fingerprint(file_names: list[str], labels: list[int]) -> str:
    raw = pcap_encoding_config_tag() + "\n"
    raw += "\n".join(f"{n}\t{l}" for n, l in zip(file_names, labels, strict=True))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_or_compute_pcap_bge_embeddings(
    file_names: list[str],
    labels: list[int],
    net_texts: list[str],
    has_pcap: list[bool] | None = None,
) -> np.ndarray:
    """按样本顺序返回 PCAP 语义 BGE 向量；无 pcap 的样本置零向量。"""
    fp = pcap_corpus_fingerprint(file_names, labels)
    if PCAP_BGE_USE_CACHE and PCAP_BGE_CACHE_PATH.is_file():
        cached = np.load(PCAP_BGE_CACHE_PATH, allow_pickle=True)
        cached_fp = cached["fingerprint"].item()
        if cached_fp == fp and len(cached["file_names"]) == len(file_names):
            log_progress(f"复用 PCAP BGE 缓存: {PCAP_BGE_CACHE_PATH.name}")
            embeddings = np.array(cached["embeddings"], dtype=np.float64)
            if has_pcap is not None:
                for idx, ok in enumerate(has_pcap):
                    if not ok:
                        embeddings[idx] = 0.0
            return embeddings

    texts_to_encode: list[str] = []
    encode_indices: list[int] = []
    dim_guess = 1024
    embeddings = np.zeros((len(net_texts), dim_guess), dtype=np.float64)
    if has_pcap is None:
        texts_to_encode = list(net_texts)
        encode_indices = list(range(len(net_texts)))
    else:
        for idx, (text, ok) in enumerate(zip(net_texts, has_pcap, strict=True)):
            if ok:
                texts_to_encode.append(text)
                encode_indices.append(idx)

    if texts_to_encode:
        log_progress(
            f"PCAP BGE 缓存未命中 (fp={fp})，分块编码 {len(texts_to_encode)}/"
            f"{len(net_texts)} 条 …"
        )
        encoded = encode_with_bge_m3(texts_to_encode)
        dim_guess = encoded.shape[1]
        if embeddings.shape[1] != dim_guess:
            embeddings = np.zeros((len(net_texts), dim_guess), dtype=np.float64)
        for local_i, global_i in enumerate(encode_indices):
            embeddings[global_i] = encoded[local_i]
    else:
        log_progress(f"PCAP BGE: 无可编码样本，全部置零")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PCAP_BGE_CACHE_PATH,
        embeddings=embeddings,
        file_names=np.array(file_names, dtype=object),
        labels=np.array(labels),
        fingerprint=np.array(fp),
    )
    log_progress(f"PCAP BGE 向量已缓存: {PCAP_BGE_CACHE_PATH}")
    return embeddings


def audit_encoding_config_tag() -> str:
    return (
        f"audit_v{BGE_ENCODING_VERSION}|pool={BGE_CHUNK_POOLING}|"
        f"chunk={BGE_CHUNK_TOKENS}|stride={BGE_CHUNK_STRIDE}|"
        f"batch={BGE_BATCH_SIZE}|dev={BGE_DEVICE}|norm={BGE_NORMALIZE}|"
        f"max_chunks={BGE_MAX_CHUNKS}|repeat={BGE_AUDIT_MAX_TOKEN_REPEAT}"
    )


def audit_corpus_fingerprint(file_names: list[str], labels: list[int]) -> str:
    raw = audit_encoding_config_tag() + "\n"
    raw += "\n".join(f"{n}\t{l}" for n, l in zip(file_names, labels, strict=True))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _apply_audit_presence_mask(
    matrix: np.ndarray,
    has_audit: list[bool] | None,
) -> np.ndarray:
    if has_audit is None:
        return matrix
    out = np.array(matrix, dtype=np.float64, copy=True)
    for idx, ok in enumerate(has_audit):
        if not ok:
            out[idx] = 0.0
    return out


def _build_audit_bge_matrix_mean(
    file_names: list[str],
    bge_audit_docs: dict[str, str],
    has_audit: list[bool] | None = None,
) -> np.ndarray:
    """mean-pool：按 file_name 读取 audit_svm 缓存，仅对缺失项补编码（不写回缓存）。"""
    cache_map = load_bge_cache_map(BGE_CACHE_PATH)
    dim = next(iter(cache_map.values())).shape[0]
    matrix = np.zeros((len(file_names), dim), dtype=np.float64)

    missing_names: list[str] = []
    missing_indices: list[int] = []
    hit = 0
    skipped = 0
    for idx, name in enumerate(file_names):
        if has_audit is not None and not has_audit[idx]:
            skipped += 1
            continue
        emb = cache_map.get(name)
        if emb is not None:
            matrix[idx] = emb
            hit += 1
        else:
            missing_names.append(name)
            missing_indices.append(idx)

    log_progress(
        f"Audit BGE(mean) 对齐: 缓存命中 {hit}/{len(file_names)}, "
        f"无audit置零 {skipped}, 缺失待编码 {len(missing_names)}"
    )
    if not missing_names:
        return matrix

    missing_texts = [bge_audit_docs.get(name, "") for name in missing_names]
    log_progress(f"对 {len(missing_names)} 个缺失样本分块编码 …")
    missing_embs = encode_with_bge_m3(missing_texts)
    if matrix.shape[1] != missing_embs.shape[1]:
        matrix = np.zeros((len(file_names), missing_embs.shape[1]), dtype=np.float64)
        for idx, name in enumerate(file_names):
            if emb := cache_map.get(name):
                matrix[idx] = emb
    for local_i, global_i in enumerate(missing_indices):
        matrix[global_i] = missing_embs[local_i]
    return matrix


def _load_or_compute_audit_bge_max(
    file_names: list[str],
    labels: list[int],
    bge_audit_docs: dict[str, str],
    has_audit: list[bool] | None = None,
) -> np.ndarray:
    """max-pool：整批 fingerprint 对齐缓存；未命中则编码并写入 audit_bge_m3_max_embeddings.npz。"""
    fp = audit_corpus_fingerprint(file_names, labels)
    if AUDIT_BGE_USE_CACHE and AUDIT_BGE_MAX_CACHE_PATH.is_file():
        cached = np.load(AUDIT_BGE_MAX_CACHE_PATH, allow_pickle=True)
        cached_fp = cached["fingerprint"].item()
        if cached_fp == fp and len(cached["file_names"]) == len(file_names):
            log_progress(
                f"复用 Audit BGE(max) 缓存: {AUDIT_BGE_MAX_CACHE_PATH.name}"
            )
            return _apply_audit_presence_mask(cached["embeddings"], has_audit)

    encode_names: list[str] = []
    encode_indices: list[int] = []
    for idx, name in enumerate(file_names):
        if has_audit is not None and not has_audit[idx]:
            continue
        encode_names.append(name)
        encode_indices.append(idx)

    log_progress(
        f"Audit BGE(max) 缓存未命中 (fp={fp})，分块编码 "
        f"{len(encode_names)}/{len(file_names)} 条 …"
    )
    matrix = np.zeros((len(file_names), 1024), dtype=np.float64)
    if encode_names:
        encode_texts = [bge_audit_docs.get(name, "") for name in encode_names]
        encoded = encode_with_bge_m3(encode_texts)
        if matrix.shape[1] != encoded.shape[1]:
            matrix = np.zeros((len(file_names), encoded.shape[1]), dtype=np.float64)
        for local_i, global_i in enumerate(encode_indices):
            matrix[global_i] = encoded[local_i]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        AUDIT_BGE_MAX_CACHE_PATH,
        embeddings=matrix,
        file_names=np.array(file_names, dtype=object),
        labels=np.array(labels),
        fingerprint=np.array(fp),
    )
    log_progress(f"Audit BGE(max) 向量已缓存: {AUDIT_BGE_MAX_CACHE_PATH}")
    return _apply_audit_presence_mask(matrix, has_audit)


def build_audit_bge_matrix(
    file_names: list[str],
    labels: list[int],
    bge_audit_docs: dict[str, str],
    has_audit: list[bool] | None = None,
) -> np.ndarray:
    """按 file_names 顺序对齐 BGE 向量；无 audit 的样本置零向量。"""
    if BGE_CHUNK_POOLING == "mean":
        return _build_audit_bge_matrix_mean(file_names, bge_audit_docs, has_audit)
    return _load_or_compute_audit_bge_max(
        file_names, labels, bge_audit_docs, has_audit
    )


def _dns_name(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def decode_hex_payload(value: str) -> str:
    try:
        return bytes.fromhex(value).decode("utf-8", errors="replace")
    except ValueError:
        return ""


def is_valid_domain(name: str) -> bool:
    name = name.strip().lower().rstrip(".")
    return bool(name) and len(name) <= 253 and bool(DOMAIN_RE.match(name))


def extract_host_side_features(audit_path: Path) -> tuple[str, np.ndarray]:
    """从 audit log 提取 TF-IDF 附加文本与主机数值特征。"""
    text = audit_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    syscall_total = syscall_failed = syscall_exit_neg2 = 0
    execve_total = execve_failed = 0
    payload_parts: list[str] = []

    for line in lines:
        if "type=SYSCALL" in line:
            syscall_total += 1
            if "success=no" in line:
                syscall_failed += 1
            m_exit = re.search(r"\bexit=(-?\d+)", line)
            if m_exit and m_exit.group(1) == "-2":
                syscall_exit_neg2 += 1
            if "SYSCALL=execve" in line or "syscall=59" in line:
                execve_total += 1
                if "success=no" in line:
                    execve_failed += 1
        elif RE_EXECVE_LINE.search(line):
            for _arg_idx, hex_val in RE_EXECVE_HEX_ARG.findall(line):
                decoded = decode_hex_payload(hex_val)
                if decoded and any(ch.isprintable() for ch in decoded):
                    payload_parts.append(decoded)

    failed_ratio = syscall_failed / syscall_total if syscall_total else 0.0
    execve_failed_ratio = execve_failed / execve_total if execve_total else 0.0
    payload_text = " ".join(payload_parts)

    numeric = np.array(
        [
            float(syscall_total),
            float(syscall_failed),
            float(syscall_exit_neg2),
            failed_ratio,
            float(execve_total),
            float(execve_failed),
            execve_failed_ratio,
            float(sum(len(p) for p in payload_parts)),
        ],
        dtype=np.float64,
    )
    return payload_text, numeric


def _open_pcap_reader(path: Path):
    import dpkt

    handle = path.open("rb")
    header = handle.read(4)
    handle.seek(0)
    if header[:4] in (
        b"\xd4\xc3\xb2\xa1",
        b"\xa1\xb2\xc3\xd4",
        b"\x4d\x3c\xb2\xa1",
        b"\xa1\xb2\x3c\x4d",
    ):
        return dpkt.pcap.Reader(handle)
    return dpkt.pcapng.Reader(handle)


def _parse_tls_sni(data: bytes) -> str | None:
    """从 TLS ClientHello 记录解析 SNI（动态步进 Session ID / CipherSuites）。"""
    if len(data) < 5 or data[0] != 0x16:
        return None
    try:
        hs = data[5:]
        if not hs or hs[0] != 0x01:
            return None
        # Handshake: type(1)+len(3)+version(2)+random(32) => Session ID len @ offset 38
        pos = 38
        if pos >= len(hs):
            return None
        sid_len = hs[pos]
        pos += 1 + sid_len
        if pos + 2 > len(hs):
            return None
        cs_len = struct.unpack("!H", hs[pos : pos + 2])[0]
        pos += 2 + cs_len
        if pos >= len(hs):
            return None
        cm_len = hs[pos]
        pos += 1 + cm_len
        if pos + 2 > len(hs):
            return None
        ext_total = struct.unpack("!H", hs[pos : pos + 2])[0]
        pos += 2
        ext_end = pos + ext_total
        while pos + 4 <= ext_end:
            etype, elen = struct.unpack("!HH", hs[pos : pos + 4])
            pos += 4
            ext_data = hs[pos : pos + elen]
            pos += elen
            if etype != 0 or len(ext_data) < 5:
                continue
            sp = 2
            while sp + 3 <= len(ext_data):
                name_type = ext_data[sp]
                name_len = struct.unpack("!H", ext_data[sp + 1 : sp + 3])[0]
                sp += 3
                if sp + name_len > len(ext_data):
                    break
                if name_type == 0:
                    candidate = ext_data[sp : sp + name_len].decode(
                        "utf-8", errors="replace"
                    ).lower()
                    if is_valid_domain(candidate):
                        return candidate
                sp += name_len
    except (struct.error, IndexError, UnicodeDecodeError):
        return None
    return None


def _tcp_flow_key(src: bytes, sport: int, dst: bytes, dport: int) -> tuple[bytes, int, bytes, int]:
    return (src, sport, dst, dport)


class _BoundedTcpStreams:
    """LRU 限流的 TCP 流缓冲，避免高并发 pcap 撑爆内存。"""

    def __init__(self, max_streams: int, max_buf: int) -> None:
        self._max_streams = max_streams
        self._max_buf = max_buf
        self._streams: OrderedDict[
            tuple[bytes, int, bytes, int], bytearray
        ] = OrderedDict()

    def append(
        self,
        key: tuple[bytes, int, bytes, int],
        data: bytes,
    ) -> bytearray:
        if key not in self._streams:
            while len(self._streams) >= self._max_streams:
                self._streams.popitem(last=False)
            self._streams[key] = bytearray()
        self._streams.move_to_end(key)
        buf = self._streams[key]
        if len(buf) >= self._max_buf:
            return buf
        room = self._max_buf - len(buf)
        buf.extend(data[:room])
        return buf

    def values(self):
        return self._streams.values()


def _append_tcp_stream(
    streams: _BoundedTcpStreams,
    key: tuple[bytes, int, bytes, int],
    data: bytes,
) -> bytearray:
    return streams.append(key, data)


def _extract_http_hosts_from_buffer(buf: bytes) -> set[str]:
    hosts: set[str] = set()
    for line in buf.split(b"\r\n"):
        if line.lower().startswith(b"host:"):
            host = line.split(b":", 1)[1].strip().decode(
                "utf-8", errors="replace"
            ).lower()
            if is_valid_domain(host):
                hosts.add(host)
    return hosts


def _private_ipv4(addr: bytes) -> bool:
    if len(addr) != 4:
        return False
    a, b, _c, _d = addr
    if a == 10:
        return True
    if a == 192 and b == 168:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 127:
        return True
    return False


def _format_ipv4(addr: bytes) -> str | None:
    if len(addr) != 4:
        return None
    return ".".join(str(b) for b in addr)


def _remote_public_ip(ip_pkt: object) -> str | None:
    """取会话对端公网 IPv4（用于无 DNS/SNI 的直连 C2 信号）。"""
    src = getattr(ip_pkt, "src", b"")
    dst = getattr(ip_pkt, "dst", b"")
    if _private_ipv4(src):
        remote = dst
    else:
        remote = src
    if _private_ipv4(remote):
        return None
    return _format_ipv4(remote)


def extract_pcap_features(pcap_path: Path) -> tuple[np.ndarray, str]:
    """从单个 PCAP 提取流统计与应用层元数据文本。"""
    import dpkt

    reader = _open_pcap_reader(pcap_path)
    lengths: list[int] = []
    iats: list[float] = []
    dns_domains: set[str] = set()
    sni_names: set[str] = set()
    http_hosts: set[str] = set()
    remote_ips: set[str] = set()
    up_bytes = 0.0
    down_bytes = 0.0
    tls_streams = _BoundedTcpStreams(MAX_TCP_STREAMS, MAX_TCP_STREAM_BUF)
    http_streams = _BoundedTcpStreams(MAX_TCP_STREAMS, MAX_TCP_STREAM_BUF)
    prev_ts: float | None = None

    for ts, buf in reader:
        lengths.append(len(buf))
        if prev_ts is not None:
            iats.append(max(float(ts - prev_ts), 0.0))
        prev_ts = float(ts)

        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except (dpkt.UnpackError, dpkt.NeedData):
            continue
        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            continue

        ip_len = float(getattr(ip, "len", len(buf)))
        if _private_ipv4(ip.src):
            up_bytes += ip_len
        else:
            down_bytes += ip_len

        remote_ip = _remote_public_ip(ip)
        if remote_ip:
            remote_ips.add(remote_ip)

        payload = ip.data
        if isinstance(payload, dpkt.udp.UDP) and (
            payload.dport == 53 or payload.sport == 53
        ):
            try:
                dns_pkt = dpkt.dns.DNS(payload.data)
                for q in dns_pkt.qd or []:
                    name = _dns_name(q.name).lower().rstrip(".")
                    if is_valid_domain(name):
                        dns_domains.add(name)
                for rr in dns_pkt.an or []:
                    if hasattr(rr, "name"):
                        name = _dns_name(rr.name).lower().rstrip(".")
                        if is_valid_domain(name):
                            dns_domains.add(name)
            except (dpkt.UnpackError, dpkt.NeedData, UnicodeDecodeError):
                pass
        elif isinstance(payload, dpkt.tcp.TCP):
            data = payload.data
            if not data:
                continue
            flow_key = _tcp_flow_key(ip.src, payload.sport, ip.dst, payload.dport)
            if payload.dport == 443:
                tls_buf = _append_tcp_stream(tls_streams, flow_key, data)
                sni = _parse_tls_sni(bytes(tls_buf))
                if sni:
                    sni_names.add(sni)
            if payload.dport in (80, 8080, 8000):
                http_buf = _append_tcp_stream(http_streams, flow_key, data)
                http_hosts.update(_extract_http_hosts_from_buffer(bytes(http_buf)))

    for tls_buf in tls_streams.values():
        if tls_buf and tls_buf[0] == 0x16:
            sni = _parse_tls_sni(bytes(tls_buf))
            if sni:
                sni_names.add(sni)
    for http_buf in http_streams.values():
        http_hosts.update(_extract_http_hosts_from_buffer(bytes(http_buf)))

    total_bytes = up_bytes + down_bytes
    up_ratio = up_bytes / total_bytes if total_bytes else 0.0
    pkt_count = float(len(lengths))

    numeric = np.array(
        [
            pkt_count,
            total_bytes,
            up_bytes,
            down_bytes,
            up_ratio,
            float(min(lengths)) if lengths else 0.0,
            float(max(lengths)) if lengths else 0.0,
            float(statistics.mean(lengths)) if lengths else 0.0,
            float(statistics.pstdev(lengths)) if len(lengths) > 1 else 0.0,
            float(statistics.mean(iats)) if iats else 0.0,
            float(statistics.pstdev(iats)) if len(iats) > 1 else 0.0,
            float(max(iats)) if iats else 0.0,
            float(len(dns_domains)),
            float(len(sni_names)),
            float(len(http_hosts)),
        ],
        dtype=np.float64,
    )

    meta_tokens: list[str] = []
    for domain in sorted(dns_domains):
        meta_tokens.append(f"dns:{domain}")
    for domain in sorted(sni_names):
        meta_tokens.append(f"sni:{domain}")
    for domain in sorted(http_hosts):
        meta_tokens.append(f"host:{domain}")
    for ip_addr in sorted(remote_ips):
        meta_tokens.append(f"ip:{ip_addr}")

    return numeric, " ".join(meta_tokens)


def _log1p_numeric(arr: np.ndarray, indices: range) -> np.ndarray:
    out = arr.copy()
    for idx in indices:
        out[idx] = np.log1p(max(out[idx], 0.0))
    return out


def transform_host_numeric(arr: np.ndarray) -> np.ndarray:
    out = _log1p_numeric(arr, range(0, 3))  # counts
    out[7] = np.log1p(max(out[7], 0.0))  # payload chars
    return out


def transform_net_numeric(arr: np.ndarray) -> np.ndarray:
    out = _log1p_numeric(arr, range(0, 4))  # packet/byte counts
    for idx in range(5, 8):
        out[idx] = np.log1p(max(out[idx], 0.0))
    out[11] = np.log1p(max(out[11], 0.0))
    for idx in range(12, 15):
        out[idx] = np.log1p(max(out[idx], 0.0))
    return out


def apply_numeric_weights(matrix: np.ndarray, mode: EvalMode) -> np.ndarray:
    """在 MinMaxScaler 之后施加模态权重（StandardScaler/MinMax 之前乘权重会被约掉）。"""
    if matrix.size == 0:
        return matrix
    out = matrix.copy()
    col = 0
    if mode in ("fusion", "audit_only"):
        n_host = len(HOST_NUMERIC_NAMES)
        out[:, col : col + n_host] *= HOST_NUMERIC_WEIGHT
        col += n_host
    if mode in ("fusion", "pcap_only"):
        n_net = len(NET_NUMERIC_NAMES)
        out[:, col : col + n_net] *= NET_NUMERIC_WEIGHT
    return out


def load_documents_from_token_csv(
    csv_path: Path, label: int, max_repeat: int | None = None
) -> tuple[dict[str, str], list[str]]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["file_name"]].append((row["token"], int(row["count"])))

    docs: dict[str, str] = {}
    names = sorted(grouped)
    for file_name in names:
        docs[file_name] = counter_to_document(grouped[file_name], max_repeat=max_repeat)
    return docs, names


def load_documents_from_audit_dir(
    audit_dir: Path, label: int, max_repeat: int | None = None
) -> tuple[dict[str, str], list[str]]:
    log_files = sorted(audit_dir.glob("*.log"))
    docs: dict[str, str] = {}
    names: list[str] = []
    total = len(log_files)
    for idx, path in enumerate(log_files, start=1):
        if idx == 1 or idx % 100 == 0 or idx == total:
            log_progress(f"分词 ({idx}/{total}): {path.name}")
        counter = count_tokens_in_audit_file(path)
        docs[path.name] = counter_to_document(list(counter.items()), max_repeat=max_repeat)
        names.append(path.name)
    return docs, names


def load_audit_docs_for_class(
    audit_dir: Path,
    token_csv: Path,
    class_name: str,
    max_repeat: int | None = None,
) -> tuple[dict[str, str], list[str]]:
    if PREFER_TOKEN_CSV and token_csv.is_file():
        log_progress(f"从 CSV 加载 {class_name} audit 语料: {token_csv.name}")
        return load_documents_from_token_csv(token_csv, LABEL_INNOCENT, max_repeat=max_repeat)
    log_progress(f"从 raw .log 加载 {class_name} audit 语料: {audit_dir}")
    return load_documents_from_audit_dir(audit_dir, LABEL_INNOCENT, max_repeat=max_repeat)


def load_pcap_features(
    innocent_names: list[str],
    malicious_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, bool]]:
    file_names = innocent_names + malicious_names
    name_to_network_dir = {
        name: INNOCENT_NETWORK_DIR for name in innocent_names
    }
    name_to_network_dir.update(
        {name: MALICIOUS_NETWORK_DIR for name in malicious_names}
    )

    try:
        import dpkt  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PCAP 特征提取需要 dpkt，请安装:\n"
            "  pip install -r scripts/requirements-network.txt"
        ) from exc

    net_numeric: dict[str, np.ndarray] = {}
    net_text: dict[str, str] = {}
    has_pcap: dict[str, bool] = {}

    total = len(file_names)
    t0 = time.perf_counter()
    for idx, audit_name in enumerate(file_names, start=1):
        pcap_name = audit_name_to_pcap_name(audit_name)
        network_dir = name_to_network_dir[audit_name]
        pcap_path = network_dir / pcap_name
        if idx == 1 or idx % 100 == 0 or idx == total:
            log_progress(f"PCAP 特征 ({idx}/{total}): {pcap_name}")
        if not pcap_path.is_file():
            net_numeric[audit_name] = np.zeros(len(NET_NUMERIC_NAMES), dtype=np.float64)
            net_text[audit_name] = ""
            has_pcap[audit_name] = False
            continue
        numeric, meta = extract_pcap_features(pcap_path)
        net_numeric[audit_name] = numeric
        net_text[audit_name] = meta
        has_pcap[audit_name] = True

    log_progress(
        f"PCAP 特征提取完成 (v{PCAP_FEATURE_VERSION}): "
        f"{sum(has_pcap.values())}/{total} 有 PCAP，"
        f"耗时 {time.perf_counter() - t0:.1f}s"
    )
    return net_numeric, net_text, has_pcap


def build_samples_for_class(
    audit_dir: Path,
    network_dir: Path,
    audit_docs: dict[str, str],
    file_names: list[str],
    label: int,
    class_name: str,
    mode: EvalMode,
    shared_pcap_numeric: dict[str, np.ndarray] | None = None,
    shared_pcap_text: dict[str, str] | None = None,
    shared_has_pcap: dict[str, bool] | None = None,
) -> list[SampleFeatures]:
    samples: list[SampleFeatures] = []
    total = len(file_names)
    need_audit = mode in ("fusion", "audit_only")
    need_pcap = mode in ("fusion", "pcap_only")
    feature_label = "主机特征" if need_audit else "网络特征"

    for idx, file_name in enumerate(file_names, start=1):
        display_name = (
            audit_name_to_pcap_name(file_name) if need_pcap and not need_audit else file_name
        )
        if idx == 1 or idx % 100 == 0 or idx == total:
            log_progress(
                f"{feature_label} ({class_name}) ({idx}/{total}): {display_name}"
            )

        if need_audit:
            audit_path = audit_dir / file_name
            has_audit = audit_path.is_file()
            if has_audit:
                payload_text, host_num = extract_host_side_features(audit_path)
                audit_text = audit_docs.get(file_name, "")
                if payload_text:
                    audit_text = f"{audit_text} {payload_text}".strip()
            else:
                audit_text = ""
                host_num = np.zeros(len(HOST_NUMERIC_NAMES), dtype=np.float64)
        else:
            audit_text = ""
            host_num = np.zeros(len(HOST_NUMERIC_NAMES), dtype=np.float64)
            has_audit = False

        if need_pcap:
            if shared_pcap_numeric is not None:
                net_num = shared_pcap_numeric.get(
                    file_name, np.zeros(len(NET_NUMERIC_NAMES), dtype=np.float64)
                )
                net_meta = shared_pcap_text.get(file_name, "") if shared_pcap_text else ""
                has_pcap = shared_has_pcap.get(file_name, False) if shared_has_pcap else False
            else:
                pcap_path = network_dir / audit_name_to_pcap_name(file_name)
                if pcap_path.is_file():
                    net_num, net_meta = extract_pcap_features(pcap_path)
                    has_pcap = True
                else:
                    net_num = np.zeros(len(NET_NUMERIC_NAMES), dtype=np.float64)
                    net_meta = ""
                    has_pcap = False
        else:
            net_num = np.zeros(len(NET_NUMERIC_NAMES), dtype=np.float64)
            net_meta = ""
            has_pcap = False

        samples.append(
            SampleFeatures(
                file_name=file_name,
                label=label,
                audit_text=audit_text,
                host_numeric=host_num,
                net_text=net_meta,
                net_numeric=net_num,
                has_audit=has_audit,
                has_pcap=has_pcap,
            )
        )
    return samples


def log_modality_coverage(samples: list[SampleFeatures], context: str) -> None:
    both = sum(1 for s in samples if s.has_audit and s.has_pcap)
    audit_only = sum(1 for s in samples if s.has_audit and not s.has_pcap)
    pcap_only = sum(1 for s in samples if s.has_pcap and not s.has_audit)
    neither = sum(1 for s in samples if not s.has_audit and not s.has_pcap)
    log_progress(
        f"{context} 模态覆盖: 总计={len(samples)}, 双模态={both}, "
        f"仅audit={audit_only}, 仅pcap={pcap_only}, 皆无={neither}"
    )


def stack_modality_flags(samples: list[SampleFeatures], mode: EvalMode) -> np.ndarray:
    """fusion 模式下追加 0/1 指示位，不参与 MinMaxScaler。"""
    if mode != "fusion":
        return np.zeros((len(samples), 0), dtype=np.float64)
    return np.array(
        [[float(s.has_audit), float(s.has_pcap)] for s in samples],
        dtype=np.float64,
    )


def stack_numeric_matrix(samples: list[SampleFeatures], mode: EvalMode) -> np.ndarray:
    rows: list[np.ndarray] = []
    for sample in samples:
        parts: list[np.ndarray] = []
        if mode in ("fusion", "audit_only"):
            parts.append(transform_host_numeric(sample.host_numeric))
        if mode in ("fusion", "pcap_only"):
            parts.append(transform_net_numeric(sample.net_numeric))
        rows.append(np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64))
    if not rows:
        return np.zeros((len(samples), 0), dtype=np.float64)
    return np.vstack(rows)


def fit_feature_artifacts(
    samples: list[SampleFeatures],
    mode: EvalMode,
    audit_encoder: SemanticEncoderName,
    net_encoder: SemanticEncoderName,
) -> FusionFeatureArtifacts:
    artifacts = FusionFeatureArtifacts(
        mode=mode,
        audit_encoder=audit_encoder,
        net_encoder=net_encoder,
    )
    if mode in ("fusion", "audit_only") and audit_encoder == "tfidf":
        audit_vec = TfidfVectorizer(
            max_features=AUDIT_TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,
            dtype=np.float64,
        )
        audit_vec.fit([s.audit_text for s in samples])
        artifacts.audit_tfidf = audit_vec

    if mode in ("fusion", "pcap_only") and net_encoder == "tfidf":
        net_vec = TfidfVectorizer(
            max_features=NET_TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=min(TFIDF_MAX_DF, 1.0),
            sublinear_tf=True,
            dtype=np.float64,
        )
        net_vec.fit([s.net_text for s in samples])
        artifacts.net_tfidf = net_vec

    x_num = stack_numeric_matrix(samples, mode)
    if x_num.shape[1] > 0:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_num)
        artifacts.numeric_scaler = scaler
    return artifacts


def transform_feature_matrix(
    samples: list[SampleFeatures],
    artifacts: FusionFeatureArtifacts,
    audit_bge_rows: np.ndarray | None = None,
    net_bge_rows: np.ndarray | None = None,
) -> csr_matrix:
    mode = artifacts.mode
    sparse_blocks: list[csr_matrix] = []

    if mode in ("fusion", "audit_only"):
        if artifacts.audit_encoder == "bge-m3":
            if audit_bge_rows is None:
                raise ValueError("audit_encoder=bge-m3 需要传入 audit_bge_rows")
            sparse_blocks.append(csr_matrix(audit_bge_rows, dtype=np.float64))
        else:
            if artifacts.audit_tfidf is None:
                raise ValueError("缺少 audit TF-IDF 变换器")
            sparse_blocks.append(
                artifacts.audit_tfidf.transform([s.audit_text for s in samples])
            )

    if mode in ("fusion", "pcap_only"):
        if artifacts.net_encoder == "bge-m3":
            if net_bge_rows is None:
                raise ValueError("net_encoder=bge-m3 需要传入 net_bge_rows")
            sparse_blocks.append(csr_matrix(net_bge_rows, dtype=np.float64))
        else:
            if artifacts.net_tfidf is None:
                raise ValueError("缺少 net TF-IDF 变换器")
            sparse_blocks.append(
                artifacts.net_tfidf.transform([s.net_text for s in samples])
            )

    x_sparse = sparse_blocks[0] if len(sparse_blocks) == 1 else hstack(
        sparse_blocks, format="csr"
    )

    x_num = stack_numeric_matrix(samples, mode)
    flags = stack_modality_flags(samples, mode)
    if artifacts.numeric_scaler is not None and x_num.shape[1] > 0:
        x_num = artifacts.numeric_scaler.transform(x_num)
        x_num = apply_numeric_weights(x_num, mode)
    if flags.shape[1] > 0:
        x_num = np.hstack([x_num, flags]) if x_num.size else flags
    if x_num.size:
        return hstack([x_sparse, csr_matrix(x_num)], format="csr")
    return x_sparse


def build_feature_blocks(
    samples: list[SampleFeatures],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mode: EvalMode,
    audit_bge_matrix: np.ndarray | None = None,
    net_bge_matrix: np.ndarray | None = None,
    audit_encoder: SemanticEncoderName | None = None,
    net_encoder: SemanticEncoderName | None = None,
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    y_train = np.array([s.label for s in train])
    y_test = np.array([s.label for s in test])

    ae = audit_encoder or AUDIT_ENCODER
    ne = net_encoder or NET_ENCODER
    artifacts = fit_feature_artifacts(train, mode, ae, ne)

    audit_train_rows = (
        audit_bge_matrix[train_idx] if audit_bge_matrix is not None else None
    )
    audit_test_rows = (
        audit_bge_matrix[test_idx] if audit_bge_matrix is not None else None
    )
    net_train_rows = net_bge_matrix[train_idx] if net_bge_matrix is not None else None
    net_test_rows = net_bge_matrix[test_idx] if net_bge_matrix is not None else None

    x_train = transform_feature_matrix(train, artifacts, audit_train_rows, net_train_rows)
    x_test = transform_feature_matrix(test, artifacts, audit_test_rows, net_test_rows)
    return x_train, x_test, y_train, y_test


def evaluate_fold(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[LABEL_INNOCENT, LABEL_MALICIOUS])
    return acc, prec, rec, f1, cm


def misclassification_type(actual: int, predicted: int) -> str:
    if actual == LABEL_INNOCENT and predicted == LABEL_MALICIOUS:
        return "FP"
    if actual == LABEL_MALICIOUS and predicted == LABEL_INNOCENT:
        return "FN"
    return "OK"


def collect_fold_misclassifications(
    samples: list[SampleFeatures],
    test_idx: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    fold_idx: int,
) -> list[dict[str, str | int | bool]]:
    rows: list[dict[str, str | int | bool]] = []
    for local_i, global_i in enumerate(test_idx):
        actual = int(y_test[local_i])
        predicted = int(y_pred[local_i])
        if actual == predicted:
            continue
        sample = samples[int(global_i)]
        rows.append(
            {
                "fold": fold_idx,
                "file_name": sample.file_name,
                "actual": LABEL_NAMES[actual],
                "predicted": LABEL_NAMES[predicted],
                "error_type": misclassification_type(actual, predicted),
                "has_audit": sample.has_audit,
                "has_pcap": sample.has_pcap,
            }
        )
    return rows


def log_fold_misclassifications(fold_idx: int, rows: list[dict[str, str | int | bool]]) -> None:
    if not rows:
        log_progress(f"[CV {fold_idx}/{N_FOLDS}] 测试集全部分对")
        return
    log_progress(f"[CV {fold_idx}/{N_FOLDS}] 分错 {len(rows)} 个样本:")
    for row in rows:
        audit_flag = "有Audit" if row["has_audit"] else "无Audit"
        pcap_flag = "有PCAP" if row["has_pcap"] else "无PCAP"
        log_progress(
            f"  - {row['file_name']}  "
            f"真实={row['actual']}  预测={row['predicted']}  "
            f"({row['error_type']}, {audit_flag}, {pcap_flag})"
        )


def summarize_unique_misclassifications(
    rows: list[dict[str, str | int | bool]],
) -> list[dict[str, str | int]]:
    """按 file_name 汇总：某样本在多少折里被分错。"""
    by_file: dict[str, dict[str, object]] = {}
    for row in rows:
        name = str(row["file_name"])
        if name not in by_file:
            by_file[name] = {
                "file_name": name,
                "actual": row["actual"],
                "wrong_folds": 0,
                "fp_count": 0,
                "fn_count": 0,
                "folds": [],
            }
        entry = by_file[name]
        entry["wrong_folds"] = int(entry["wrong_folds"]) + 1
        if row["error_type"] == "FP":
            entry["fp_count"] = int(entry["fp_count"]) + 1
        elif row["error_type"] == "FN":
            entry["fn_count"] = int(entry["fn_count"]) + 1
        folds = entry["folds"]
        assert isinstance(folds, list)
        folds.append(int(row["fold"]))

    summary: list[dict[str, str | int]] = []
    for entry in sorted(
        by_file.values(),
        key=lambda x: (-int(x["wrong_folds"]), str(x["file_name"])),
    ):
        folds = entry["folds"]
        assert isinstance(folds, list)
        summary.append(
            {
                "file_name": str(entry["file_name"]),
                "actual": str(entry["actual"]),
                "wrong_folds": int(entry["wrong_folds"]),
                "fp_count": int(entry["fp_count"]),
                "fn_count": int(entry["fn_count"]),
                "fold_list": ",".join(str(f) for f in sorted(folds)),
            }
        )
    return summary


def append_fold_row(
    fold_rows: list[dict[str, float | int]],
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    y_test: np.ndarray,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    cm: np.ndarray,
    elapsed: float,
) -> None:
    y_train = y[train_idx]
    fold_rows.append(
        {
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_innocent": int(np.sum(y_train == LABEL_INNOCENT)),
            "train_malicious": int(np.sum(y_train == LABEL_MALICIOUS)),
            "test_innocent": int(np.sum(y_test == LABEL_INNOCENT)),
            "test_malicious": int(np.sum(y_test == LABEL_MALICIOUS)),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "fit_predict_sec": round(elapsed, 2),
        }
    )


def run_stratified_kfold_cv(
    samples: list[SampleFeatures],
    mode: EvalMode,
    audit_bge_matrix: np.ndarray | None = None,
    net_bge_matrix: np.ndarray | None = None,
) -> tuple[list[dict[str, float | int]], np.ndarray, list[dict[str, str | int | bool]]]:
    y = np.array([s.label for s in samples])
    groups = build_cv_groups_from_samples(samples)
    validate_group_cv(groups, y, N_FOLDS)
    sgkf = make_group_stratified_kfold()
    fold_rows: list[dict[str, float | int]] = []
    misclassified_rows: list[dict[str, str | int | bool]] = []
    conf_sum = np.zeros((2, 2), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(np.zeros(len(samples)), y, groups=groups), start=1
    ):
        assert_no_group_leakage(groups, train_idx, test_idx)
        log_progress(
            f"[CV {fold_idx}/{N_FOLDS}] 训练 {len(train_idx)} / 测试 {len(test_idx)}"
        )
        x_train, x_test, y_train, y_test = build_feature_blocks(
            samples, train_idx, test_idx, mode, audit_bge_matrix, net_bge_matrix
        )

        svm = LinearSVC(
            C=SVM_C,
            class_weight=SVM_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            dual="auto",
        )
        t0 = time.perf_counter()
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        elapsed = time.perf_counter() - t0

        acc, prec, rec, f1, cm = evaluate_fold(y_test, y_pred)
        conf_sum += cm
        fold_mis = collect_fold_misclassifications(
            samples, test_idx, y_test, y_pred, fold_idx
        )
        misclassified_rows.extend(fold_mis)
        log_fold_misclassifications(fold_idx, fold_mis)
        append_fold_row(
            fold_rows,
            fold_idx,
            train_idx,
            test_idx,
            y,
            y_test,
            acc,
            prec,
            rec,
            f1,
            cm,
            elapsed,
        )
        log_progress(
            f"[CV {fold_idx}/{N_FOLDS}] acc={acc:.4f} f1={f1:.4f} "
            f"prec={prec:.4f} rec={rec:.4f} ({elapsed:.1f}s)"
        )
    return fold_rows, conf_sum, misclassified_rows


def model_config_dict(
    mode: EvalMode,
    audit_encoder: SemanticEncoderName,
    net_encoder: SemanticEncoderName,
    train_sample_count: int,
) -> dict[str, object]:
    return {
        "bundle_version": MODEL_BUNDLE_VERSION,
        "mode": mode,
        "audit_encoder": audit_encoder,
        "net_encoder": net_encoder,
        "train_sample_count": train_sample_count,
        "require_both_modalities": REQUIRE_BOTH_MODALITIES,
        "label_names": LABEL_NAMES,
        "host_numeric_names": HOST_NUMERIC_NAMES,
        "net_numeric_names": NET_NUMERIC_NAMES,
        "modality_flag_names": MODALITY_FLAG_NAMES,
        "audit_tfidf_max_features": AUDIT_TFIDF_MAX_FEATURES,
        "net_tfidf_max_features": NET_TFIDF_MAX_FEATURES,
        "tfidf_min_df": TFIDF_MIN_DF,
        "tfidf_max_df": TFIDF_MAX_DF,
        "host_numeric_weight": HOST_NUMERIC_WEIGHT,
        "net_numeric_weight": NET_NUMERIC_WEIGHT,
        "pcap_feature_version": PCAP_FEATURE_VERSION,
        "bge_model": BGE_MODEL,
        "bge_chunk_tokens": BGE_CHUNK_TOKENS,
        "bge_chunk_stride": BGE_CHUNK_STRIDE,
        "bge_chunk_pooling": BGE_CHUNK_POOLING,
        "bge_batch_size": BGE_BATCH_SIZE,
        "bge_normalize": BGE_NORMALIZE,
        "bge_encoding_version": BGE_ENCODING_VERSION,
        "pcap_bge_encoding_version": PCAP_BGE_ENCODING_VERSION,
        "max_token_repeat": MAX_TOKEN_REPEAT,
        "svm_c": SVM_C,
        "svm_class_weight": SVM_CLASS_WEIGHT,
        "random_state": RANDOM_STATE,
    }


def save_fusion_model_bundle(
    model_dir: Path,
    svm: LinearSVC,
    artifacts: FusionFeatureArtifacts,
    mode: EvalMode,
    audit_encoder: SemanticEncoderName,
    net_encoder: SemanticEncoderName,
    train_sample_count: int,
) -> None:
    import joblib

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config = model_config_dict(
        mode, audit_encoder, net_encoder, train_sample_count
    )
    (model_dir / "model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(svm, model_dir / "svm.joblib")
    joblib.dump(artifacts, model_dir / "feature_artifacts.joblib")
    log_progress(f"模型已保存: {model_dir}")


def load_fusion_model_bundle(
    model_dir: Path,
) -> tuple[LinearSVC, FusionFeatureArtifacts, dict[str, object]]:
    import joblib

    model_dir = Path(model_dir)
    config_path = model_dir / "model_config.json"
    if not config_path.is_file():
        raise SystemExit(f"模型配置不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("bundle_version") != MODEL_BUNDLE_VERSION:
        log_progress(
            f"警告: 模型 bundle 版本 {config.get('bundle_version')!r} "
            f"与当前代码 {MODEL_BUNDLE_VERSION!r} 不一致"
        )
    svm = joblib.load(model_dir / "svm.joblib")
    artifacts = joblib.load(model_dir / "feature_artifacts.joblib")
    return svm, artifacts, config


def load_fusion_samples(
    run_mode: RunMode = "fusion",
    audit_encoder: SemanticEncoderName | None = None,
    net_encoder: SemanticEncoderName | None = None,
) -> tuple[
    list[SampleFeatures],
    np.ndarray | None,
    np.ndarray | None,
    SemanticEncoderName,
    SemanticEncoderName,
]:
    """加载样本并（按需）预计算 BGE 矩阵。"""
    ae = audit_encoder or AUDIT_ENCODER
    ne = net_encoder or NET_ENCODER
    run_mode = validate_run_mode(run_mode)

    if run_mode == "all":
        need_audit = need_pcap = True
        build_mode: EvalMode = "fusion"
    elif run_mode == "fusion":
        need_audit = need_pcap = True
        build_mode = "fusion"
    elif run_mode == "audit_only":
        need_audit, need_pcap = True, False
        build_mode = "audit_only"
    else:
        need_audit, need_pcap = False, True
        build_mode = "pcap_only"

    inn_names, mal_names, _ = resolve_sample_names(
        run_mode, USE_UNIFIED_SAMPLES, REQUIRE_BOTH_MODALITIES
    )
    if MAX_SAMPLES_PER_CLASS > 0:
        inn_names = inn_names[:MAX_SAMPLES_PER_CLASS]
        mal_names = mal_names[:MAX_SAMPLES_PER_CLASS]

    if need_audit:
        inn_docs, _ = load_audit_docs_for_class(
            INNOCENT_AUDIT_DIR, INNOCENT_TOKEN_CSV, "innocent"
        )
        mal_docs, _ = load_audit_docs_for_class(
            MALICIOUS_AUDIT_DIR, MALICIOUS_TOKEN_CSV, "malicious"
        )
    else:
        inn_docs = mal_docs = {}

    bge_audit_docs: dict[str, str] = {}
    if need_audit and ae == "bge-m3":
        inn_bge, _ = load_audit_docs_for_class(
            INNOCENT_AUDIT_DIR,
            INNOCENT_TOKEN_CSV,
            "innocent (BGE)",
            max_repeat=BGE_AUDIT_MAX_TOKEN_REPEAT,
        )
        mal_bge, _ = load_audit_docs_for_class(
            MALICIOUS_AUDIT_DIR,
            MALICIOUS_TOKEN_CSV,
            "malicious (BGE)",
            max_repeat=BGE_AUDIT_MAX_TOKEN_REPEAT,
        )
        bge_audit_docs = {**inn_bge, **mal_bge}

    if need_pcap:
        net_numeric_map, net_text_map, has_pcap_map = load_pcap_features(
            inn_names, mal_names
        )
    else:
        net_numeric_map = net_text_map = has_pcap_map = None

    inn_samples = build_samples_for_class(
        INNOCENT_AUDIT_DIR,
        INNOCENT_NETWORK_DIR,
        inn_docs,
        inn_names,
        LABEL_INNOCENT,
        "innocent",
        build_mode,
        net_numeric_map,
        net_text_map,
        has_pcap_map,
    )
    mal_samples = build_samples_for_class(
        MALICIOUS_AUDIT_DIR,
        MALICIOUS_NETWORK_DIR,
        mal_docs,
        mal_names,
        LABEL_MALICIOUS,
        "malicious",
        build_mode,
        net_numeric_map,
        net_text_map,
        has_pcap_map,
    )
    samples = inn_samples + mal_samples

    audit_bge_matrix: np.ndarray | None = None
    if need_audit and ae == "bge-m3":
        file_names = [s.file_name for s in samples]
        has_audit_flags = [s.has_audit for s in samples]
        audit_bge_matrix = build_audit_bge_matrix(
            file_names,
            [s.label for s in samples],
            bge_audit_docs,
            has_audit_flags,
        )

    net_bge_matrix: np.ndarray | None = None
    if need_pcap and ne == "bge-m3":
        file_names = [s.file_name for s in samples]
        net_bge_matrix = load_or_compute_pcap_bge_embeddings(
            file_names,
            [s.label for s in samples],
            [s.net_text for s in samples],
            [s.has_pcap for s in samples],
        )

    return samples, audit_bge_matrix, net_bge_matrix, ae, ne


def train_fusion_model(
    samples: list[SampleFeatures],
    mode: EvalMode,
    audit_encoder: SemanticEncoderName,
    net_encoder: SemanticEncoderName,
    audit_bge_matrix: np.ndarray | None = None,
    net_bge_matrix: np.ndarray | None = None,
) -> tuple[LinearSVC, FusionFeatureArtifacts]:
    log_progress(
        f"全量训练 fusion 模型: n={len(samples)}, mode={mode}, "
        f"audit={audit_encoder}, net={net_encoder}"
    )
    artifacts = fit_feature_artifacts(samples, mode, audit_encoder, net_encoder)
    x = transform_feature_matrix(
        samples, artifacts, audit_bge_matrix, net_bge_matrix
    )
    y = np.array([s.label for s in samples])
    svm = LinearSVC(
        C=SVM_C,
        class_weight=SVM_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        dual="auto",
    )
    svm.fit(x, y)
    log_progress(f"训练完成: X.shape={x.shape}")
    return svm, artifacts


def predict_fusion_samples(
    samples: list[SampleFeatures],
    svm: LinearSVC,
    artifacts: FusionFeatureArtifacts,
    audit_bge_matrix: np.ndarray | None = None,
    net_bge_matrix: np.ndarray | None = None,
) -> np.ndarray:
    x = transform_feature_matrix(
        samples, artifacts, audit_bge_matrix, net_bge_matrix
    )
    return svm.predict(x)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    acc, prec, rec, f1, cm = evaluate_fold(y_true, y_pred)
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }, cm


def write_prediction_csv(
    path: Path,
    samples: list[SampleFeatures],
    y_pred: np.ndarray,
    y_true: np.ndarray | None = None,
) -> None:
    rows: list[dict[str, str | int | bool]] = []
    for idx, sample in enumerate(samples):
        row: dict[str, str | int | bool] = {
            "file_name": sample.file_name,
            "predicted": LABEL_NAMES[int(y_pred[idx])],
            "predicted_label": int(y_pred[idx]),
            "has_audit": sample.has_audit,
            "has_pcap": sample.has_pcap,
        }
        if y_true is not None:
            row["actual"] = LABEL_NAMES[int(y_true[idx])]
            row["actual_label"] = int(y_true[idx])
            row["correct"] = int(y_true[idx]) == int(y_pred[idx])
        rows.append(row)
    fields = list(rows[0].keys())
    write_csv(path, rows, fields)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(
    mode: EvalMode,
    samples: list[SampleFeatures],
    inn_count: int,
    mal_count: int,
    pcap_hits: int,
    summary_rows: list[dict],
    conf_sum: np.ndarray,
    cv_path: Path,
    summary_path: Path,
    conf_path: Path,
    misclassified_path: Path,
    misclassified_rows: list[dict[str, str | int | bool]],
    audit_encoder: SemanticEncoderName,
    net_encoder: SemanticEncoderName,
) -> None:
    mode_title = {
        "fusion": "Audit + PCAP 早期融合",
        "audit_only": "仅 Audit 特征",
        "pcap_only": "仅 PCAP 特征",
    }[mode]
    print(f"\n=== {mode_title} + LinearSVC 十折交叉验证 ===")
    print(f"RUN_MODE: {mode}")
    print(
        f"样本数: {len(samples)} (innocent={inn_count}, malicious={mal_count}, "
        f"有 PCAP={pcap_hits})"
    )
    if mode in ("fusion", "audit_only"):
        print(
            f"Audit 语义: {audit_encoder}"
            + (
                f" (TF-IDF max_features={AUDIT_TFIDF_MAX_FEATURES})"
                if audit_encoder == "tfidf"
                else (
                    f" (BGE-M3 pool={BGE_CHUNK_POOLING}, cache="
                    f"{(BGE_CACHE_PATH if BGE_CHUNK_POOLING == 'mean' else AUDIT_BGE_MAX_CACHE_PATH).name}, "
                    f"chunk_tokens={BGE_CHUNK_TOKENS}, batch={BGE_BATCH_SIZE})"
                )
            )
        )
    if mode in ("fusion", "pcap_only"):
        print(
            f"Network 语义: {net_encoder}"
            + (
                f" (TF-IDF max_features={NET_TFIDF_MAX_FEATURES})"
                if net_encoder == "tfidf"
                else (
                    f" (BGE-M3 cache={PCAP_BGE_CACHE_PATH.name}, "
                    f"chunk_tokens={BGE_CHUNK_TOKENS}, batch={BGE_BATCH_SIZE})"
                )
            )
        )
    print(
        f"数值特征: log1p + MinMaxScaler(0,1) + 模态权重; "
        f"host_weight={HOST_NUMERIC_WEIGHT}, net_weight={NET_NUMERIC_WEIGHT}"
    )
    print(f"SVM: LinearSVC(C={SVM_C}, class_weight={SVM_CLASS_WEIGHT!r})")
    print(
        f"CV: StratifiedGroupKFold(n_splits={N_FOLDS}, 按 skill 分组, "
        f"random_state={RANDOM_STATE})"
    )
    print(f"\n{'metric':<12} {'mean':>8} {'std':>8}")
    for row in summary_rows:
        print(f"{row['metric']:<12} {row['mean']:>8.4f} {row['std']:>8.4f}")

    print("\n十折汇总混淆矩阵 (行=真实, 列=预测):")
    print("              innocent  malicious")
    print(
        f"  innocent    {conf_sum[0,0]:8d}  {conf_sum[0,1]:8d}\n"
        f"  malicious   {conf_sum[1,0]:8d}  {conf_sum[1,1]:8d}"
    )
    print(f"\n每折结果: {cv_path}")
    print(f"汇总统计: {summary_path}")
    print(f"混淆矩阵: {conf_path}")
    print(f"错分明细: {misclassified_path}")

    unique_wrong = summarize_unique_misclassifications(misclassified_rows)
    print(f"\n错分样本汇总: 共 {len(misclassified_rows)} 条错分记录（跨 {N_FOLDS} 折），"
          f"涉及 {len(unique_wrong)} 个不同文件")
    if unique_wrong:
        print(f"{'file_name':<55} {'真实':<10} {'错分折数':>8}  {'类型':<8}  折号")
        for row in unique_wrong:
            err = []
            if int(row["fp_count"]) > 0:
                err.append(f"FP×{row['fp_count']}")
            if int(row["fn_count"]) > 0:
                err.append(f"FN×{row['fn_count']}")
            print(
                f"{row['file_name']:<55} {row['actual']:<10} "
                f"{row['wrong_folds']:>8}  {','.join(err):<8}  {row['fold_list']}"
            )
    else:
        print("  （全部样本在所有折中均分类正确）")


def run_single_eval(
    eval_mode: EvalMode,
    samples: list[SampleFeatures],
    inn_count: int,
    mal_count: int,
    pcap_hits: int,
    audit_encoder: SemanticEncoderName,
    net_encoder: SemanticEncoderName,
    audit_bge_matrix: np.ndarray | None = None,
    net_bge_matrix: np.ndarray | None = None,
) -> dict[str, float | str]:
    cv_path, summary_path, conf_path, misclassified_path = output_paths(eval_mode)

    log_progress(f"开始 {N_FOLDS} 折分组交叉验证 ({eval_mode}, StratifiedGroupKFold) …")
    fold_rows, conf_sum, misclassified_rows = run_stratified_kfold_cv(
        samples, eval_mode, audit_bge_matrix, net_bge_matrix
    )

    metrics = ["accuracy", "precision", "recall", "f1"]
    summary_rows: list[dict[str, float | str]] = []
    for metric in metrics:
        vals = [float(row[metric]) for row in fold_rows]
        summary_rows.append(
            {
                "metric": metric,
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
        )

    write_csv(cv_path, fold_rows, list(fold_rows[0].keys()))
    write_csv(summary_path, summary_rows, ["metric", "mean", "std", "min", "max"])
    write_csv(
        conf_path,
        [
            {
                "actual": LABEL_NAMES[LABEL_INNOCENT],
                "predict_innocent": int(conf_sum[0, 0]),
                "predict_malicious": int(conf_sum[0, 1]),
            },
            {
                "actual": LABEL_NAMES[LABEL_MALICIOUS],
                "predict_innocent": int(conf_sum[1, 0]),
                "predict_malicious": int(conf_sum[1, 1]),
            },
        ],
        ["actual", "predict_innocent", "predict_malicious"],
    )
    write_csv(
        misclassified_path,
        misclassified_rows,
        ["fold", "file_name", "actual", "predicted", "error_type", "has_audit", "has_pcap"],
    )

    print_report(
        eval_mode,
        samples,
        inn_count,
        mal_count,
        pcap_hits,
        summary_rows,
        conf_sum,
        cv_path,
        summary_path,
        conf_path,
        misclassified_path,
        misclassified_rows,
        audit_encoder,
        net_encoder,
    )

    return {
        "mode": eval_mode,
        **{str(row["metric"]): float(row["mean"]) for row in summary_rows},
    }


def print_ablation_summary(
    results: list[dict[str, float | str]],
    sample_count: int,
    inn_count: int,
    mal_count: int,
) -> None:
    print("\n=== 公平 Ablation 汇总（统一样本交集） ===")
    print(f"样本数: {sample_count} (innocent={inn_count}, malicious={mal_count})")
    print(f"{'mode':<12} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for row in results:
        print(
            f"{row['mode']:<12} "
            f"{float(row['accuracy']):>10.4f} "
            f"{float(row['precision']):>10.4f} "
            f"{float(row['recall']):>10.4f} "
            f"{float(row['f1']):>10.4f}"
        )


def configure_runtime(
    *,
    data_root: Path | str,
    output_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    run_mode: RunMode | None = None,
    audit_encoder: SemanticEncoderName | None = None,
    net_encoder: SemanticEncoderName | None = None,
    require_both_modalities: bool | None = None,
) -> None:
    """配置数据根目录、结果目录与 BGE 缓存目录（换数据集时调用）。"""
    global INNOCENT_AUDIT_DIR, MALICIOUS_AUDIT_DIR
    global INNOCENT_NETWORK_DIR, MALICIOUS_NETWORK_DIR
    global INNOCENT_TOKEN_CSV, MALICIOUS_TOKEN_CSV
    global COMPARISON_DIR, OUTPUT_DIR
    global UNIFIED_MANIFEST_PATH, FUSION_MANIFEST_PATH
    global BGE_CACHE_PATH, AUDIT_BGE_MAX_CACHE_PATH, PCAP_BGE_CACHE_PATH
    global RUN_MODE, AUDIT_ENCODER, NET_ENCODER, REQUIRE_BOTH_MODALITIES

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"数据根目录不存在: {root}")

    inn_root = root / "innocent-by-type"
    mal_root = root / "malicious-by-type"
    INNOCENT_AUDIT_DIR = inn_root / "audit-logs"
    MALICIOUS_AUDIT_DIR = mal_root / "audit-logs"
    INNOCENT_NETWORK_DIR = inn_root / "network"
    MALICIOUS_NETWORK_DIR = mal_root / "network"
    INNOCENT_TOKEN_CSV = inn_root / "audit_token_per_file.csv"
    MALICIOUS_TOKEN_CSV = mal_root / "audit_token_per_file.csv"

    COMPARISON_DIR = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else root / "comparison"
    )
    OUTPUT_DIR = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else COMPARISON_DIR / "audit_tfidf_pcap_encoder"
    )
    UNIFIED_MANIFEST_PATH = OUTPUT_DIR / "fusion_svm_unified_sample_manifest.csv"
    FUSION_MANIFEST_PATH = OUTPUT_DIR / "fusion_svm_fusion_sample_manifest.csv"

    BGE_CACHE_PATH = COMPARISON_DIR / "audit_bge_m3_embeddings.npz"
    AUDIT_BGE_MAX_CACHE_PATH = COMPARISON_DIR / "audit_bge_m3_max_embeddings.npz"
    PCAP_BGE_CACHE_PATH = COMPARISON_DIR / "pcap_bge_m3_embeddings.npz"

    if run_mode is not None:
        RUN_MODE = run_mode
    if audit_encoder is not None:
        AUDIT_ENCODER = audit_encoder
    if net_encoder is not None:
        NET_ENCODER = net_encoder
    if require_both_modalities is not None:
        REQUIRE_BOTH_MODALITIES = require_both_modalities

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def parse_cli_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit TF-IDF + PCAP BGE-M3 早期融合 + LinearSVC 十折分组交叉验证",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="数据根目录（含 innocent-by-type/ 与 malicious-by-type/）；"
        "默认使用脚本上级目录（开发模式）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果 CSV 输出目录（默认 <data-root>/comparison/audit_tfidf_pcap_encoder）",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="BGE 编码缓存目录（默认 <data-root>/comparison）",
    )
    parser.add_argument(
        "--run-mode",
        choices=["fusion", "audit_only", "pcap_only", "all"],
        default=None,
        help="运行模式（默认 fusion）",
    )
    parser.add_argument(
        "--audit-encoder",
        choices=["tfidf", "bge-m3"],
        default=None,
        help="Audit 语义编码（默认 tfidf）",
    )
    parser.add_argument(
        "--net-encoder",
        choices=["tfidf", "bge-m3"],
        default=None,
        help="PCAP 语义编码（默认 bge-m3）",
    )
    return parser.parse_args()


def main() -> None:
    run_mode = validate_run_mode(RUN_MODE)
    audit_encoder = validate_audit_encoder(AUDIT_ENCODER)
    net_encoder = validate_net_encoder(NET_ENCODER)
    log_progress(
        f"RUN_MODE={run_mode!r}, AUDIT_ENCODER={audit_encoder!r}, "
        f"NET_ENCODER={net_encoder!r}, output={OUTPUT_DIR}"
    )
    t0 = time.perf_counter()

    inn_names, mal_names, unified = resolve_sample_names(
        run_mode, USE_UNIFIED_SAMPLES, REQUIRE_BOTH_MODALITIES
    )

    if MAX_SAMPLES_PER_CLASS > 0:
        inn_names = inn_names[:MAX_SAMPLES_PER_CLASS]
        mal_names = mal_names[:MAX_SAMPLES_PER_CLASS]

    if unified:
        write_unified_manifest(inn_names, mal_names)
    elif run_mode == "fusion":
        write_fusion_manifest(
            inn_names,
            mal_names,
            INNOCENT_AUDIT_DIR,
            MALICIOUS_AUDIT_DIR,
            INNOCENT_NETWORK_DIR,
            MALICIOUS_NETWORK_DIR,
        )

    if run_mode == "all":
        eval_modes: list[EvalMode] = ["fusion", "audit_only", "pcap_only"]
        need_audit = need_pcap = True
    elif run_mode == "fusion":
        eval_modes = ["fusion"]
        need_audit = need_pcap = True
    elif run_mode == "audit_only":
        eval_modes = ["audit_only"]
        need_audit = True
        need_pcap = False
    else:
        eval_modes = ["pcap_only"]
        need_audit = False
        need_pcap = True

    if need_audit:
        inn_docs, _ = load_audit_docs_for_class(
            INNOCENT_AUDIT_DIR, INNOCENT_TOKEN_CSV, "innocent"
        )
        mal_docs, _ = load_audit_docs_for_class(
            MALICIOUS_AUDIT_DIR, MALICIOUS_TOKEN_CSV, "malicious"
        )
    else:
        log_progress("pcap_only: 不读取 audit 日志/CSV")
        inn_docs = mal_docs = {}

    audit_bge_matrix: np.ndarray | None = None
    bge_audit_docs: dict[str, str] = {}
    if need_audit and audit_encoder == "bge-m3":
        inn_bge_docs, _ = load_audit_docs_for_class(
            INNOCENT_AUDIT_DIR,
            INNOCENT_TOKEN_CSV,
            "innocent (BGE repeat)",
            max_repeat=BGE_AUDIT_MAX_TOKEN_REPEAT,
        )
        mal_bge_docs, _ = load_audit_docs_for_class(
            MALICIOUS_AUDIT_DIR,
            MALICIOUS_TOKEN_CSV,
            "malicious (BGE repeat)",
            max_repeat=BGE_AUDIT_MAX_TOKEN_REPEAT,
        )
        bge_audit_docs = {**inn_bge_docs, **mal_bge_docs}

    if need_pcap:
        net_numeric_map, net_text_map, has_pcap_map = load_pcap_features(
            inn_names, mal_names
        )
    else:
        log_progress("audit_only: 不读取 network/*.pcap")
        net_numeric_map = net_text_map = has_pcap_map = None

    build_mode: EvalMode = (
        "fusion" if need_audit and need_pcap else eval_modes[0]
    )
    log_progress(f"构建样本 (build_mode={build_mode}) …")
    inn_samples = build_samples_for_class(
        INNOCENT_AUDIT_DIR,
        INNOCENT_NETWORK_DIR,
        inn_docs,
        inn_names,
        LABEL_INNOCENT,
        "innocent",
        build_mode,
        net_numeric_map,
        net_text_map,
        has_pcap_map,
    )
    mal_samples = build_samples_for_class(
        MALICIOUS_AUDIT_DIR,
        MALICIOUS_NETWORK_DIR,
        mal_docs,
        mal_names,
        LABEL_MALICIOUS,
        "malicious",
        build_mode,
        net_numeric_map,
        net_text_map,
        has_pcap_map,
    )
    samples = inn_samples + mal_samples
    pcap_hits = sum(1 for s in samples if s.has_pcap)
    audit_hits = sum(1 for s in samples if s.has_audit)

    log_progress(
        f"样本就绪: innocent={len(inn_samples)}, malicious={len(mal_samples)}, "
        f"有 Audit={audit_hits}/{len(samples)}, 有 PCAP={pcap_hits}/{len(samples)}，"
        f"耗时 {time.perf_counter() - t0:.1f}s"
    )
    if build_mode == "fusion":
        log_modality_coverage(samples, "fusion")

    empty_audit = sum(1 for s in samples if not s.audit_text.strip())
    if empty_audit and need_audit and audit_encoder == "tfidf":
        log_progress(f"警告: {empty_audit} 个样本 audit 文本为空")

    if need_audit and audit_encoder == "bge-m3":
        file_names = [s.file_name for s in samples]
        has_audit_flags = [s.has_audit for s in samples]
        audit_bge_matrix = build_audit_bge_matrix(
            file_names,
            [s.label for s in samples],
            bge_audit_docs,
            has_audit_flags,
        )
        log_progress(f"Audit BGE 特征矩阵: {audit_bge_matrix.shape}")

    net_bge_matrix: np.ndarray | None = None
    if need_pcap and net_encoder == "bge-m3":
        file_names = [s.file_name for s in samples]
        labels = [s.label for s in samples]
        net_texts = [s.net_text for s in samples]
        has_pcap_flags = [s.has_pcap for s in samples]
        net_bge_matrix = load_or_compute_pcap_bge_embeddings(
            file_names, labels, net_texts, has_pcap_flags
        )
        log_progress(f"PCAP BGE 特征矩阵: {net_bge_matrix.shape}")

    empty_net = sum(1 for s in samples if not s.net_text.strip())
    if empty_net and need_pcap and net_encoder == "tfidf":
        log_progress(f"警告: {empty_net} 个样本 network 文本为空")

    ablation_results: list[dict[str, float | str]] = []
    for idx, eval_mode in enumerate(eval_modes):
        if run_mode == "all" and idx > 0:
            log_progress(f"--- ablation: {eval_mode} ---")
        ablation_results.append(
            run_single_eval(
                eval_mode,
                samples,
                len(inn_samples),
                len(mal_samples),
                pcap_hits,
                audit_encoder,
                net_encoder,
                audit_bge_matrix if eval_mode in ("fusion", "audit_only") else None,
                net_bge_matrix if eval_mode in ("fusion", "pcap_only") else None,
            )
        )

    if run_mode == "all":
        print_ablation_summary(
            ablation_results,
            len(samples),
            len(inn_samples),
            len(mal_samples),
        )
        log_progress(
            f"全部 ablation 完成，总耗时 {time.perf_counter() - t0:.1f}s"
        )


if __name__ == "__main__":
    args = parse_cli_args()
    if args.data_root is not None:
        configure_runtime(
            data_root=args.data_root,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            run_mode=args.run_mode,  # type: ignore[arg-type]
            audit_encoder=args.audit_encoder,  # type: ignore[arg-type]
            net_encoder=args.net_encoder,  # type: ignore[arg-type]
        )
    main()
