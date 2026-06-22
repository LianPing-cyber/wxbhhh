#!/usr/bin/env python3
"""在训练集上拟合模型并导出 model_bundle/（仅维护者运行一次）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent / "lib"
sys.path.insert(0, str(LIB_DIR))

import audit_pcap_fusion_svm_classifier as pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并导出 Audit-TF-IDF + PCAP-BGE fusion 模型")
    parser.add_argument("--data-root", type=Path, required=True, help="训练数据根目录")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="模型输出目录（默认 <本目录>/model）",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="BGE 缓存目录")
    parser.add_argument("--audit-encoder", default="tfidf", choices=["tfidf", "bge-m3"])
    parser.add_argument("--net-encoder", default="bge-m3", choices=["tfidf", "bge-m3"])
    parser.add_argument(
        "--require-both-modalities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅保留 audit 与 pcap 均存在的样本（默认开启）",
    )
    args = parser.parse_args()

    pkg = Path(__file__).resolve().parent
    model_dir = args.model_dir or (pkg / "model")

    pipeline.configure_runtime(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        require_both_modalities=args.require_both_modalities,
    )
    t0 = time.perf_counter()
    samples, audit_bge, net_bge, ae, ne = pipeline.load_fusion_samples(
        run_mode="fusion",
        audit_encoder=args.audit_encoder,  # type: ignore[arg-type]
        net_encoder=args.net_encoder,  # type: ignore[arg-type]
    )
    svm, artifacts = pipeline.train_fusion_model(
        samples, "fusion", ae, ne, audit_bge, net_bge
    )
    pipeline.save_fusion_model_bundle(
        model_dir,
        svm,
        artifacts,
        "fusion",
        ae,
        ne,
        len(samples),
    )
    print(f"导出完成: {model_dir}，样本数={len(samples)}，耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
