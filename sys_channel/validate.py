#!/usr/bin/env python3
"""
加载已训练模型，在新数据上做验证/推理（不训练）。

数据目录与训练时相同（innocent-by-type / malicious-by-type），
若两类都有标签则输出准确率等指标；也可只做预测 CSV。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LIB_DIR = Path(__file__).resolve().parent / "lib"
sys.path.insert(0, str(LIB_DIR))

import audit_pcap_fusion_svm_classifier as pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit TF-IDF + PCAP BGE-M3 融合模型 — 验证/推理"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="验证数据根目录（含 innocent-by-type/ 与 malicious-by-type/）",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="模型目录（默认 <本目录>/model）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果输出目录（默认 <data-root>/comparison/validation_results）",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="BGE 编码缓存目录（默认 <data-root>/comparison）",
    )
    parser.add_argument(
        "--require-both-modalities",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="仅评估 audit 与 pcap 均存在的样本（默认跟随 model_config.json）",
    )
    args = parser.parse_args()

    pkg = Path(__file__).resolve().parent
    model_dir = args.model_dir or (pkg / "model")
    output_dir = args.output_dir or (
        args.data_root / "comparison" / "validation_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    svm, artifacts, config = pipeline.load_fusion_model_bundle(model_dir)
    mode = config["mode"]  # type: ignore[index]
    ae = config["audit_encoder"]  # type: ignore[index]
    ne = config["net_encoder"]  # type: ignore[index]
    require_both = (
        args.require_both_modalities
        if args.require_both_modalities is not None
        else bool(config.get("require_both_modalities", True))
    )

    pipeline.configure_runtime(
        data_root=args.data_root,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        require_both_modalities=require_both,
    )

    pipeline.log_progress(f"加载模型: {model_dir}")
    pipeline.log_progress(f"bundle={config.get('bundle_version')}, mode={mode}")
    t0 = time.perf_counter()

    samples, audit_bge, net_bge, _, _ = pipeline.load_fusion_samples(
        run_mode=mode,  # type: ignore[arg-type]
        audit_encoder=ae,  # type: ignore[arg-type]
        net_encoder=ne,  # type: ignore[arg-type]
    )
    y_pred = pipeline.predict_fusion_samples(
        samples, svm, artifacts, audit_bge, net_bge
    )
    y_true = np.array([s.label for s in samples])

    pred_path = output_dir / "predictions.csv"
    pipeline.write_prediction_csv(pred_path, samples, y_pred, y_true)

    metrics, cm = pipeline.evaluate_predictions(y_true, y_pred)
    metrics_path = output_dir / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pipeline.write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {
                "actual": pipeline.LABEL_NAMES[pipeline.LABEL_INNOCENT],
                "predict_innocent": int(cm[0, 0]),
                "predict_malicious": int(cm[0, 1]),
            },
            {
                "actual": pipeline.LABEL_NAMES[pipeline.LABEL_MALICIOUS],
                "predict_innocent": int(cm[1, 0]),
                "predict_malicious": int(cm[1, 1]),
            },
        ],
        ["actual", "predict_innocent", "predict_malicious"],
    )

    wrong = int(np.sum(y_true != y_pred))
    pipeline.log_progress(
        f"验证完成: n={len(samples)}, 错分={wrong}, "
        f"acc={metrics['accuracy']}, f1={metrics['f1']}, 耗时 {time.perf_counter()-t0:.1f}s"
    )
    print(f"\n预测明细: {pred_path}")
    print(f"指标 JSON: {metrics_path}")
    print(f"混淆矩阵: {output_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
