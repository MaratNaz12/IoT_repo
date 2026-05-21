"""Массовый запуск 4 LLM-скриптов в двух режимах по всем CSV.

Прогоняет cгенерированные LLM-пайплайны (Gemini en/ru, GPT en/ru) в
двух конфигурациях (без IP / с IP) по всем CSV из указанной папки,
кроме исключённых датасетов.

Структура артефактов:
    <results-dir>/<имя_варианта>/<имя_csv_без_расширения>/
        - metrics.csv, summary.txt, *_confusion_matrix.png,
          *_classification_report.txt, best_model.joblib,
          eda/, features.json, run.log
    <results-dir>/run_log.csv  — общая сводка по запускам

Примеры использования:
    python run_all.py
    python run_all.py --data-dir data_sampled --results-dir results
    python run_all.py --datasets CTU-IoT-Malware-Capture-20-1conn.log.labeled.csv
    python run_all.py --variants gemini_en gpt_en
    python run_all.py --skip-existing
    python run_all.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = PROJECT_ROOT / "llm_generated"

EXCLUDED_FILES = {
    "CTU-IoT-Malware-Capture-21-1conn.log.labeled.csv",
    "CTU-IoT-Malware-Capture-60-1conn.log.labeled.csv",
}

# (имя варианта, имя файла в llm_generated/, доп. аргументы)
VARIANTS: list[tuple[str, str, list[str]]] = [
    ("gemini_en",         "gemini_3-1_pro_generated_en.py", []),
    ("gemini_en_ip",      "gemini_3-1_pro_generated_en.py", ["--include-ip", "--test-size", "0.3"]),
    ("gemini_ru_no_ip",   "gemini_3-1_pro_generated_ru.py", []),
    ("gemini_ru_with_ip", "gemini_3-1_pro_generated_ru.py", ["--include-ip"]),
    ("gpt_en",            "gpt_5-5_generated_en.py",        []),
    ("gpt_en_ip",         "gpt_5-5_generated_en.py",        ["--include-ip"]),
    ("gpt_ru",            "gpt_5-5_generated_ru.py",        []),
    ("gpt_ru_ip",         "gpt_5-5_generated_ru.py",        ["--include-ip"]),
]

SUMMARY_HEADER = ["dataset", "variant", "status", "elapsed_sec",
                  "returncode", "output_dir"]


def collect_datasets(data_dir: Path, only: list[str] | None) -> list[Path]:
    if only:
        files = [data_dir / name for name in only]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                "Не найдены файлы: " + ", ".join(str(m) for m in missing)
            )
        return files
    return sorted(
        p for p in data_dir.glob("*.csv") if p.name not in EXCLUDED_FILES
    )


def select_variants(only: list[str] | None) -> list[tuple[str, str, list[str]]]:
    if not only:
        return VARIANTS
    wanted = set(only)
    unknown = wanted - {v[0] for v in VARIANTS}
    if unknown:
        raise ValueError(f"Неизвестные варианты: {sorted(unknown)}")
    return [v for v in VARIANTS if v[0] in wanted]


def run_one(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    """Запускает один подпроцесс, пишет stdout/stderr в log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env
        )
    return proc.returncode, time.time() - start


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Имена конкретных CSV (по умолчанию: все, кроме исключённых)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help=f"Какие варианты запускать (default: все 8). "
             f"Доступны: {', '.join(v[0] for v in VARIANTS)}",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Пропускать варианты, у которых уже есть metrics.csv",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Останавливаться при первой ошибке",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Только вывести команды, ничего не запускать",
    )
    parser.add_argument(
        "--show-warnings", action="store_true",
        help="Не глушить предупреждения в подзапусках (по умолчанию подавляются)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_dir.is_dir():
        print(f"[!] Папка с данными не найдена: {args.data_dir.resolve()}")
        return 1

    try:
        datasets = collect_datasets(args.data_dir, args.datasets)
        variants = select_variants(args.variants)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[!] {exc}")
        return 1

    if not datasets:
        print(f"[!] В {args.data_dir} нет CSV для обработки")
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "run_log.csv"
    new_summary = not summary_path.exists()

    child_env = os.environ.copy()
    if not args.show_warnings:
        child_env["PYTHONWARNINGS"] = "ignore"

    total = len(datasets) * len(variants)
    print(f"Датасетов: {len(datasets)}, вариантов: {len(variants)}, "
          f"всего запусков: {total}")
    print(f"Исключены: {sorted(EXCLUDED_FILES)}")
    if not args.show_warnings:
        print("Предупреждения в подзапусках подавлены (PYTHONWARNINGS=ignore).")
    print()

    overall_start = time.time()
    n_ok = n_fail = n_skip = 0

    with summary_path.open("a", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        if new_summary:
            writer.writerow(SUMMARY_HEADER)

        idx = 0
        for ds in datasets:
            ds_stem = ds.name.removesuffix(".csv")
            for variant_name, script, extra in variants:
                idx += 1
                out_dir = args.results_dir / variant_name / ds_stem
                log_path = out_dir / "run.log"
                cmd = [
                    sys.executable, str(LLM_DIR / script),
                    "--data-path", str(ds),
                    "--output-dir", str(out_dir),
                    *extra,
                ]
                header = (
                    f"[{idx:>3}/{total}] {now()}  "
                    f"{variant_name:<18} | {ds.name}"
                )

                if args.skip_existing and (out_dir / "metrics.csv").exists():
                    print(f"{header}  -> SKIP (уже посчитано)")
                    writer.writerow([ds.name, variant_name, "skip", 0, 0,
                                     str(out_dir)])
                    n_skip += 1
                    continue

                if args.dry_run:
                    printable = " ".join(shlex.quote(c) for c in cmd)
                    print(f"{header}\n    $ {printable}")
                    continue

                print(f"{header}  -> START", flush=True)
                rc, elapsed = run_one(cmd, log_path, child_env)
                status = "ok" if rc == 0 else "fail"
                if rc == 0:
                    n_ok += 1
                    print(f" -> OK  ({elapsed:6.1f} s)", flush=True)
                else:
                    n_fail += 1
                    print(
                        f" -> FAIL (rc={rc}, {elapsed:6.1f} s)  "
                        f"см. {log_path}",
                        flush=True,
                    )

                writer.writerow([ds.name, variant_name, status,
                                 f"{elapsed:.2f}", rc, str(out_dir)])
                sf.flush()

                if rc != 0 and args.stop_on_error:
                    print("[!] Останавливаюсь по --stop-on-error")
                    return 2

    mins, secs = divmod(int(time.time() - overall_start), 60)
    print(f"\nИтого: OK={n_ok}, FAIL={n_fail}, SKIP={n_skip}, "
          f"время: {mins} мин {secs} сек")
    print(f"Сводка: {summary_path}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
