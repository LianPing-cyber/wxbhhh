#!/usr/bin/env python3
"""对 audit-logs 目录下 Linux audit.log 做分词，统计各词元出现次数及跨文件分布。"""

from __future__ import annotations

import csv
import re
import sqlite3
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 批量分析目录（可按需修改）
AUDIT_LOG_DIR = PROJECT_ROOT / "innocent-by-type" / "audit-logs"

# 词频 CSV 输出路径（默认与 audit-logs 同级）
OUTPUT_DIR = AUDIT_LOG_DIR.parent
OUTPUT_CSV = OUTPUT_DIR / "audit_token_counts.csv"
OUTPUT_DIST_STATS_CSV = OUTPUT_DIR / "audit_token_distribution_stats.csv"
OUTPUT_PER_FILE_CSV = OUTPUT_DIR / "audit_token_per_file.csv"
SQLITE_PATH = OUTPUT_DIR / ".audit_token_counts.sqlite"

# === 运行配置（改这里即可，无需命令行）===
# "all"      → [1/3 分词] + [2/3 统计] + [3/3 写入]
# "tokenize" → 仅 [1/3 分词]（生成/覆盖 SQLite 与 per_file CSV）
# "stats"    → 仅 [2/3 统计] + [3/3 写入]（复用已有 SQLite）
RunStage = Literal["all", "tokenize", "stats"]
RUN_STAGE: RunStage = "all"

# 全流程（RUN_STAGE="all"）结束后是否保留 SQLite
KEEP_SQLITE_AFTER_FULL_RUN = True

# key=value；值可为引号串、括号串或非空白
RE_KV = re.compile(r'\b([a-zA-Z][\w-]*)=("[^"]*"|\([^)]*\)|\S+)')

# 不作为独立词元的字段（容器）
SKIP_KV_KEYS = frozenset({"msg"})

# 统计阶段跳过的词元前缀（audit 参数项，噪声大）
SKIP_TOKEN_PREFIXES = ("a0=", "a1=", "a2=", "a3=", "pid=")

DIST_STAT_FIELDS = [
    "token",
    "total_count",
    "avg_per_file",
    "file_count",
    "files_with_token",
    "files_without_token",
    "presence_ratio",
    "min_per_file",
    "p25_per_file",
    "median_per_file",
    "p75_per_file",
    "max_per_file",
    "std_per_file",
    "mean_per_file",
]


def should_skip_token(token: str) -> bool:
    """是否跳过该词元（a0/a1/a2 参数项）。"""
    return any(token.startswith(prefix) for prefix in SKIP_TOKEN_PREFIXES)


def skip_token_sql_clause() -> tuple[str, list[str]]:
    """生成 SQL 过滤子句，排除 SKIP_TOKEN_PREFIXES。"""
    clause = " AND ".join("token NOT LIKE ?" for _ in SKIP_TOKEN_PREFIXES)
    params = [f"{prefix}%" for prefix in SKIP_TOKEN_PREFIXES]
    return clause, params


def log_progress(msg: str) -> None:
    """输出带时间戳的进度信息，flush 确保长任务中即时可见。"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def kv_token(key: str, raw_val: str) -> str:
    """生成完整的 key=value 词元；原值带引号则保留引号。"""
    key_l = key.lower()
    val = raw_val.strip().rstrip("'")
    if val.startswith('"') and val.endswith('"'):
        inner = val[1:-1].lower()
        return f'{key_l}="{inner}"'
    return f"{key_l}={val.lower()}"


def tokenize_audit_line(line: str) -> list[str]:
    """从一行 audit 记录提取分词列表（保序、可重复）。"""
    line = line.strip()
    if not line:
        return []

    tokens: list[str] = []
    for m in RE_KV.finditer(line):
        key_l = m.group(1).lower()
        if key_l in SKIP_KV_KEYS:
            continue
        tokens.append(kv_token(m.group(1), m.group(2)))
    return tokens


def normalize_audit_text(raw: str) -> str:
    """audit 行内常用 \\x1d 分隔补充字段，避免被 splitlines 误切成多行。"""
    return raw.replace("\x1d", " ")


def tokenize_audit_file(path: Path) -> list[str]:
    """对单个 audit.log 文件分词，返回全部词元（含重复）。"""
    text = normalize_audit_text(
        path.read_text(encoding="utf-8", errors="replace")
    )
    tokens: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        tokens.extend(tokenize_audit_line(line))
    return tokens


def count_tokens_in_audit_file(path: Path) -> Counter[str]:
    """逐行分词并直接累计 Counter，与 Counter(tokenize_audit_file(path)) 等价。"""
    counter: Counter[str] = Counter()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = normalize_audit_text(raw_line)
            if not line.strip():
                continue
            for token in tokenize_audit_line(line):
                counter[token] += 1
    return counter


def percentile(values: list[float], p: float) -> float:
    """线性插值百分位数。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def distribution_row_from_counts(
    token: str, nonzero_counts: list[int], file_count: int
) -> dict[str, str | int | float]:
    """由稀疏非零计数还原分布统计（与稠密 list 版数学等价）。"""
    zeros = file_count - len(nonzero_counts)
    vals = [0.0] * zeros + [float(c) for c in nonzero_counts]
    total = int(sum(vals))
    files_with_token = len(nonzero_counts)
    files_without_token = zeros
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if file_count > 1 else 0.0

    return {
        "token": token,
        "total_count": total,
        "avg_per_file": round(total / file_count, 4) if file_count else 0.0,
        "file_count": file_count,
        "files_with_token": files_with_token,
        "files_without_token": files_without_token,
        "presence_ratio": round(files_with_token / file_count, 4) if file_count else 0.0,
        "min_per_file": int(min(vals)),
        "p25_per_file": round(percentile(vals, 25), 4),
        "median_per_file": round(statistics.median(vals), 4),
        "p75_per_file": round(percentile(vals, 75), 4),
        "max_per_file": int(max(vals)),
        "std_per_file": round(std, 4),
        "mean_per_file": round(mean, 4),
    }


def init_sparse_db(db_path: Path) -> sqlite3.Connection:
    """创建用于稀疏逐文件词频的 SQLite 临时库。"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE per_file_counts (
            file_name TEXT NOT NULL,
            token TEXT NOT NULL,
            count INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX idx_per_file_counts_token ON per_file_counts(token)"
    )
    return conn


def process_logs_streaming(
    audit_dir: Path,
    out_dir: Path,
    out_per_file_csv: Path,
) -> tuple[Counter[str], int, sqlite3.Connection, Path]:
    """逐文件分词、写稀疏明细，并用 SQLite 暂存跨文件计数。"""
    if not audit_dir.is_dir():
        raise SystemExit(f"目录不存在: {audit_dir}")

    log_files = sorted(audit_dir.glob("*.log"))
    if not log_files:
        raise SystemExit(f"目录下无 .log 文件: {audit_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = SQLITE_PATH
    if db_path.exists():
        db_path.unlink()
    for suffix in ("-shm", "-wal"):
        sidecar = Path(f"{db_path}{suffix}")
        if sidecar.exists():
            sidecar.unlink()

    conn = init_sparse_db(db_path)
    total_counter: Counter[str] = Counter()
    total_files = len(log_files)
    per_file_rows = 0

    log_progress(f"[1/3 分词] 开始处理 {total_files} 个 .log 文件（流式 + 稀疏存储）…")
    t0 = time.perf_counter()

    out_per_file_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_per_file_csv.open("w", encoding="utf-8", newline="") as pf_handle:
        pf_writer = csv.writer(pf_handle)
        pf_writer.writerow(["file_name", "token", "count"])

        for idx, path in enumerate(log_files, start=1):
            log_progress(f"[1/3 分词] ({idx}/{total_files}) {path.name}")
            file_counter = count_tokens_in_audit_file(path)
            total_counter.update(file_counter)

            batch = [
                (path.name, token, count)
                for token, count in file_counter.items()
            ]
            conn.executemany(
                "INSERT INTO per_file_counts (file_name, token, count) VALUES (?, ?, ?)",
                batch,
            )
            pf_writer.writerows(batch)
            per_file_rows += len(batch)

            if idx % 20 == 0:
                conn.commit()

            del file_counter

    conn.commit()
    log_progress(
        f"[1/3 分词] 完成，耗时 {time.perf_counter() - t0:.1f}s，"
        f"去重词元 {len(total_counter)}，稀疏明细 {per_file_rows} 行"
    )
    return total_counter, per_file_rows, conn, db_path


def open_existing_db(db_path: Path) -> sqlite3.Connection:
    """打开已有 SQLite（只读）。"""
    if not db_path.is_file():
        raise SystemExit(f"SQLite 不存在: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def get_file_count_from_db(conn: sqlite3.Connection) -> int:
    """从 SQLite 读取参与统计的文件数。"""
    row = conn.execute(
        "SELECT COUNT(DISTINCT file_name) FROM per_file_counts"
    ).fetchone()
    return int(row[0]) if row else 0


def build_token_counts_from_db(conn: sqlite3.Connection) -> Counter[str]:
    """从 SQLite 聚合总词频（跳过 a0/a1/a2）。"""
    clause, params = skip_token_sql_clause()
    cursor = conn.execute(
        f"""
        SELECT token, SUM(count)
        FROM per_file_counts
        WHERE {clause}
        GROUP BY token
        """,
        params,
    )
    return Counter({token: int(total) for token, total in cursor})


def build_distribution_stats_from_db(
    conn: sqlite3.Connection, file_count: int
) -> list[dict[str, str | int | float]]:
    """单次扫描 SQLite，按词元计算跨文件分布统计。"""
    skip_desc = ", ".join(SKIP_TOKEN_PREFIXES)
    log_progress(f"[2/3 统计] 扫描稀疏计数并计算分布（跳过 {skip_desc}）…")
    t0 = time.perf_counter()

    clause, params = skip_token_sql_clause()
    cursor = conn.execute(
        f"""
        SELECT token, count
        FROM per_file_counts
        WHERE {clause}
        ORDER BY token
        """,
        params,
    )

    rows: list[dict[str, str | int | float]] = []
    current_token: str | None = None
    current_counts: list[int] = []
    token_idx = 0

    for token, count in cursor:
        if current_token is None:
            current_token = token
        if token != current_token:
            token_idx += 1
            if token_idx == 1 or token_idx % 5000 == 0:
                log_progress(
                    f"[2/3 统计] ({token_idx}) 当前: {current_token[:60]}"
                )
            rows.append(
                distribution_row_from_counts(
                    current_token, current_counts, file_count
                )
            )
            current_token = token
            current_counts = [count]
        else:
            current_counts.append(count)

    if current_token is not None:
        token_idx += 1
        rows.append(
            distribution_row_from_counts(current_token, current_counts, file_count)
        )

    rows.sort(key=lambda r: (-int(r["total_count"]), str(r["token"])))
    log_progress(
        f"[2/3 统计] 完成，耗时 {time.perf_counter() - t0:.1f}s，"
        f"词元数 {len(rows)}"
    )
    return rows


def write_token_counts_csv(
    counter: Counter[str], out_path: Path, file_count: int
) -> None:
    """将词频写入 CSV：token, count, avg_per_file（按 count 降序）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count", "avg_per_file"])
        for token, count in counter.most_common():
            avg = count / file_count if file_count else 0.0
            writer.writerow([token, count, f"{avg:.4f}"])


def write_distribution_stats_csv(
    rows: list[dict[str, str | int | float]], out_path: Path
) -> None:
    """写入各词元跨文件出现次数的分布统计。"""
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DIST_STAT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_stats_only(
    out_dir: Path,
    db_path: Path = SQLITE_PATH,
) -> tuple[Counter[str], list[dict[str, str | int | float]], int, Path, Path]:
    """从已有 SQLite 直接执行 [2/3 统计] 与 [3/3 写入]。"""
    out_csv = out_dir / "audit_token_counts.csv"
    out_dist_stats_csv = out_dir / "audit_token_distribution_stats.csv"

    log_progress(f"复用已有 SQLite: {db_path}")
    run_t0 = time.perf_counter()

    conn = open_existing_db(db_path)
    try:
        file_count = get_file_count_from_db(conn)
        log_progress(f"SQLite 中文件数: {file_count}")

        log_progress("[2/3 统计] 聚合总词频 …")
        counter = build_token_counts_from_db(conn)
        dist_rows = build_distribution_stats_from_db(conn, file_count)

        log_progress(f"[3/3 写入] 写入总词频 CSV → {out_csv.name}")
        write_token_counts_csv(counter, out_csv, file_count)
        log_progress(f"[3/3 写入] 写入分布统计 CSV → {out_dist_stats_csv.name}")
        write_distribution_stats_csv(dist_rows, out_dist_stats_csv)
    finally:
        conn.close()

    log_progress(f"统计完成，总耗时 {time.perf_counter() - run_t0:.1f}s")
    return counter, dist_rows, file_count, out_csv, out_dist_stats_csv


def run_tokenize_only(
    audit_dir: Path,
    out_dir: Path,
    out_per_file_csv: Path,
) -> tuple[Counter[str], int, int]:
    """仅执行 [1/3 分词]，保留 SQLite 供后续 stats 阶段使用。"""
    counter, per_file_rows, conn, db_path = process_logs_streaming(
        audit_dir, out_dir, out_per_file_csv
    )
    file_count = get_file_count_from_db(conn)
    conn.close()
    log_progress(f"SQLite 已保存: {db_path}")
    return counter, per_file_rows, file_count


def validate_run_stage(stage: str) -> RunStage:
    if stage not in ("all", "tokenize", "stats"):
        raise SystemExit(
            f'RUN_STAGE 无效: {stage!r}，请设为 "all" | "tokenize" | "stats"'
        )
    return stage  # type: ignore[return-value]


def main() -> None:
    stage = validate_run_stage(RUN_STAGE)
    audit_dir = AUDIT_LOG_DIR
    out_dir = OUTPUT_DIR

    out_csv = OUTPUT_CSV
    out_dist_stats_csv = OUTPUT_DIST_STATS_CSV
    out_per_file_csv = OUTPUT_PER_FILE_CSV

    log_progress(f"运行阶段: RUN_STAGE={stage!r}")
    log_progress(f"输出目录: {out_dir}")

    counter: Counter[str] | None = None
    dist_rows: list[dict[str, str | int | float]] = []
    file_count = 0
    per_file_rows: int | None = None

    if stage == "stats":
        if not SQLITE_PATH.is_file():
            raise SystemExit(f"RUN_STAGE='stats' 但未找到 SQLite: {SQLITE_PATH}")
        counter, dist_rows, file_count, out_csv, out_dist_stats_csv = run_stats_only(
            out_dir
        )
    elif stage == "tokenize":
        run_t0 = time.perf_counter()
        counter, per_file_rows, file_count = run_tokenize_only(
            audit_dir, out_dir, out_per_file_csv
        )
        log_progress(f"分词完成，总耗时 {time.perf_counter() - run_t0:.1f}s")
    else:
        log_progress(f"开始处理 audit 日志: {audit_dir}")
        run_t0 = time.perf_counter()

        counter, per_file_rows, conn, db_path = process_logs_streaming(
            audit_dir, out_dir, out_per_file_csv
        )
        file_count = get_file_count_from_db(conn)

        try:
            dist_rows = build_distribution_stats_from_db(conn, file_count)

            log_progress(f"[3/3 写入] 写入总词频 CSV → {out_csv.name}")
            write_token_counts_csv(counter, out_csv, file_count)
            log_progress(f"[3/3 写入] 写入分布统计 CSV → {out_dist_stats_csv.name}")
            write_distribution_stats_csv(dist_rows, out_dist_stats_csv)
        finally:
            conn.close()
            if KEEP_SQLITE_AFTER_FULL_RUN:
                log_progress(f"SQLite 已保留: {db_path}")
            elif db_path.exists():
                db_path.unlink()
                log_progress(f"已清理临时数据库: {db_path.name}")

        log_progress(f"全部完成，总耗时 {time.perf_counter() - run_t0:.1f}s")

    if counter is None:
        raise SystemExit("未产生统计结果，请检查 RUN_STAGE 配置")

    total = sum(counter.values())
    unique = len(counter)
    print(f"\n=== audit-logs 词频统计 ({audit_dir.name}) ===")
    print(f"文件数: {file_count}")
    print(f"总词数（含重复）: {total}，去重后: {unique}")
    print(f"已跳过词元前缀: {', '.join(SKIP_TOKEN_PREFIXES)}")
    print(f"SQLite: {SQLITE_PATH} ({'存在' if SQLITE_PATH.is_file() else '不存在'})")

    if stage != "tokenize":
        print(f"\nTop 20 词元:")
        for token, count in counter.most_common(20):
            avg = count / file_count if file_count else 0.0
            print(f"  {count:>8}  {avg:>8.2f}  {token}")

        example_token = "subj=unconfined"
        example = next((r for r in dist_rows if r["token"] == example_token), None)
        if example:
            print(f"\n示例分布 ({example_token}):")
            print(
                f"  出现在 {example['files_with_token']}/{file_count} 个文件，"
                f"总次数 {example['total_count']}"
            )
            print(
                f"  每文件次数: min={example['min_per_file']}, "
                f"p25={example['p25_per_file']}, median={example['median_per_file']}, "
                f"p75={example['p75_per_file']}, max={example['max_per_file']}, "
                f"mean={example['mean_per_file']}, std={example['std_per_file']}"
            )

        print(f"\n总词频已写入: {out_csv}")
        print(f"跨文件分布统计已写入: {out_dist_stats_csv}")

    if per_file_rows is not None:
        print(f"逐文件明细（稀疏）已写入: {out_per_file_csv} ({per_file_rows} 行)")
    elif out_per_file_csv.exists():
        print(f"逐文件明细（已有）: {out_per_file_csv}")


if __name__ == "__main__":
    main()
