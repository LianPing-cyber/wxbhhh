#!/bin/sh
# 从仓库 scripts/ 同步到交付包 lib/（改主脚本后执行一次）
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cp "$ROOT/../scripts/audit_pcap_fusion_svm_classifier.py" "$ROOT/lib/"
cp "$ROOT/../scripts/tokenize_audit_log.py" "$ROOT/lib/"
echo "已同步 lib/"
