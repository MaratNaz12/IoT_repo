"""Сэмплирование CSV для борьбы с OOM.

LLM-скрипты на больших файлах (сотни МБ — гигабайты) падают по памяти,
потому что используют плотный OneHotEncoder. Чтобы прогон оставался
сопоставимым по всем датасетам без модификации сгенерированного кода,
тяжёлые CSV предварительно урезаются стратифицированной выборкой
по колонке `label` (Benign / Malicious).

Правила обработки:
    size_mb < SIZE_THRESHOLD_MB      -> копирование 1:1 (мелкие файлы)
    size_mb >= SIZE_THRESHOLD_MB     -> чтение целиком + стратификация
    size_mb >= CHUNKED_THRESHOLD_MB  -> чтение чанками + стратификация

Пример использования:
    python sample_data.py
    python sample_data.py --dst data_sampled_20k --max-rows 20000 --size-threshold-mb 5
    python sample_data.py --files CTU-IoT-Malware-Capture-35-1conn.log.labeled.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_COLUMN = "label"
DEFAULT_CHUNK_SIZE = 200_000


def stratified_sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Стратифицированное сэмплирование по `label`, in-memory.

    Сохраняет пропорции классов как в исходном df.
    """
    if len(df) <= max_rows:
        return df

    if LABEL_COLUMN not in df.columns:
        return df.sample(n=max_rows, random_state=seed)

    classes = df[LABEL_COLUMN].value_counts()
    n_classes = len(classes)
    parts: list[pd.DataFrame] = []
    remaining = max_rows
    share = max_rows / len(df)
    for i, (cls, cnt) in enumerate(classes.items()):
        if i == n_classes - 1:
            target = max(1, min(remaining, cnt))
        else:
            target = max(1, min(int(round(cnt * share)), cnt))
        remaining -= target
        parts.append(df[df[LABEL_COLUMN] == cls].sample(n=target, random_state=seed))

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def chunked_stratified_sample(
    src: Path, max_rows: int, seed: int, chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[pd.DataFrame, int]:
    """Двупроходное сэмплирование без загрузки файла целиком.

    Pass 1: считает количество строк каждого класса.
    Pass 2: отбирает нужное число строк каждого класса.
    """
    counts: dict[str, int] = {}
    total = 0
    for chunk in pd.read_csv(src, dtype=str, chunksize=chunk_size, low_memory=False):
        total += len(chunk)
        vc = chunk[LABEL_COLUMN].value_counts()
        for cls, cnt in vc.items():
            counts[cls] = counts.get(cls, 0) + int(cnt)

    if total <= max_rows:
        return pd.read_csv(src, dtype=str, low_memory=False), total

    share = max_rows / total
    targets = {cls: max(1, int(round(cnt * share))) for cls, cnt in counts.items()}

    rng = np.random.default_rng(seed)
    collected: dict[str, list[pd.DataFrame]] = {cls: [] for cls in counts}
    taken: dict[str, int] = {cls: 0 for cls in counts}

    for chunk in pd.read_csv(src, dtype=str, chunksize=chunk_size, low_memory=False):
        for cls, target in targets.items():
            need = target - taken[cls]
            if need <= 0:
                continue
            cls_part = chunk[chunk[LABEL_COLUMN] == cls]
            if cls_part.empty:
                continue
            take = min(len(cls_part), need)
            picked = cls_part.sample(n=take, random_state=int(rng.integers(2**31)))
            collected[cls].append(picked)
            taken[cls] += take

    parts = [p for cls_parts in collected.values() for p in cls_parts]
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, total


def process_file(
    src: Path, dst: Path, max_rows: int,
    size_threshold_mb: float, chunked_threshold_mb: float,
    seed: int, overwrite: bool,
) -> str:
    """Обрабатывает один CSV: копирует / сэмплирует in-memory / сэмплирует чанками."""
    if dst.exists() and not overwrite:
        return "skip"

    size_mb = src.stat().st_size / (1024 * 1024)
    if size_mb < size_threshold_mb:
        shutil.copy2(src, dst)
        return f"copy ({size_mb:.1f} MB)"

    if size_mb >= chunked_threshold_mb:
        sampled, before = chunked_stratified_sample(src, max_rows, seed)
        sampled.to_csv(dst, index=False)
        return f"sample-chunked {before} -> {len(sampled)} ({size_mb:.1f} MB)"

    df = pd.read_csv(src, dtype=str, low_memory=False)
    before = len(df)
    sampled = stratified_sample(df, max_rows, seed)
    sampled.to_csv(dst, index=False)
    return f"sample {before} -> {len(sampled)} ({size_mb:.1f} MB)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dst", type=Path, default=PROJECT_ROOT / "data_sampled")
    parser.add_argument(
        "--max-rows", type=int, default=200_000,
        help="Сколько строк оставить в сэмпле тяжёлых CSV (default: 200000)",
    )
    parser.add_argument(
        "--size-threshold-mb", type=float, default=50.0,
        help="Файлы крупнее этого порога сэмплируются (default: 50 MB)",
    )
    parser.add_argument(
        "--chunked-threshold-mb", type=float, default=300.0,
        help="Файлы крупнее этого читаются чанками (default: 300 MB)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Конкретные имена CSV (по умолчанию все из --src)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.src.is_dir():
        print(f"[!] Папка-источник не найдена: {args.src.resolve()}")
        return 1

    args.dst.mkdir(parents=True, exist_ok=True)

    if args.files:
        files = [args.src / name for name in args.files]
        missing = [f for f in files if not f.exists()]
        if missing:
            print("[!] Не найдены:", ", ".join(str(m) for m in missing))
            return 1
    else:
        files = sorted(args.src.glob("*.csv"))

    if not files:
        print(f"[!] В {args.src} нет CSV")
        return 1

    print(f"Источник : {args.src}")
    print(f"Назначение: {args.dst}")
    print(f"Порог    : {args.size_threshold_mb:.1f} MB, max_rows={args.max_rows}\n")

    width = len(str(len(files)))
    for i, src in enumerate(files, 1):
        dst = args.dst / src.name
        try:
            status = process_file(
                src, dst, args.max_rows, args.size_threshold_mb,
                args.chunked_threshold_mb, args.seed, args.overwrite,
            )
            print(f"  {i:>{width}}/{len(files)} [OK] {src.name}  -> {status}")
        except MemoryError:
            print(
                f"  {i:>{width}}/{len(files)} [FAIL] {src.name}: не хватает памяти."
                f"Нужно понизить --max-rows"
            )
        except Exception as exc:
            print(f"  {i:>{width}}/{len(files)} [FAIL] {src.name}: {exc}")

    print("\nГотово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
