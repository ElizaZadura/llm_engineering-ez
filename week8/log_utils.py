import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

# Foreground colors
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

# Background color
BG_BLACK = '\033[40m'
BG_BLUE = '\033[44m'

# Reset code to return to default color
RESET = '\033[0m'

mapper = {
    BG_BLACK+RED: "#dd0000",
    BG_BLACK+GREEN: "#00dd00",
    BG_BLACK+YELLOW: "#dddd00",
    BG_BLACK+BLUE: "#0000ee",
    BG_BLACK+MAGENTA: "#aa00dd",
    BG_BLACK+CYAN: "#00dddd",
    BG_BLACK+WHITE: "#87CEEB",
    BG_BLUE+WHITE: "#ff7800"
}


def reformat(message):
    for key, value in mapper.items():
        message = message.replace(key, f'<span style="color: {value}">')
    message = message.replace(RESET, '</span>')
    return message


def _extract_title(doc: str) -> str:
    """Pull the first line's title text out of a Chroma document string."""
    first_line = doc.split("\n")[0]
    return first_line.replace("Title:", "").strip()


def select_informative_similars(similars, prices, k=5):
    """
    From a larger retrieved set (e.g., 50), pick a smaller subset that preserves
    semantic closeness while adding price-range coverage.

    Keep in sync with RAG prompting (e.g. messages_for in day2.ipynb).
    """
    n = len(similars)
    if n <= k:
        return similars, prices

    indices = list(range(n))
    by_price = sorted(indices, key=lambda i: prices[i])

    tail_size = max(2, n // 3)
    low_pool = by_price[:tail_size]
    high_pool = by_price[-tail_size:]

    mid = n // 2
    median_pool = by_price[max(0, mid - 2) : min(n, mid + 3)]

    selected = []
    count_total = k

    def pick(pool, count):
        for i in sorted(pool):
            if i not in selected:
                selected.append(i)
                if len(selected) >= count_total:
                    return
                count -= 1
                if count == 0:
                    return

    pick(low_pool, 2)
    pick(high_pool, 2)
    pick(median_pool, 1)

    for i in indices:
        if i not in selected:
            selected.append(i)
            if len(selected) == k:
                break

    selected = sorted(selected[:k])
    return [similars[i] for i in selected], [prices[i] for i in selected]


def _build_record(
    i,
    tester,
    test_items,
    find_similars_fn,
    run: str,
    prompt_k: int,
    select_fn,
) -> dict:
    item = test_items[i]
    documents, prices = find_similars_fn(item)
    prompt_docs, prompt_prices = select_fn(documents, prices, prompt_k)
    return {
        "title": item.title,
        "predicted": round(tester.guesses[i], 2),
        "actual": round(tester.truths[i], 2),
        "error": round(tester.errors[i], 2),
        "prompt_titles": [_extract_title(d) for d in prompt_docs],
        "prompt_prices": [round(p, 2) for p in prompt_prices],
        "run": run,
    }


def log_predictions(
    tester,
    test_items,
    find_similars_fn,
    run: str,
    k: int = 5,
    log_dir: str = "logs",
    prompt_k: int = 5,
    select_fn=None,
    use_timestamp_subdir: bool = True,
) -> tuple[str, str, str]:
    """
    Write the k worst and k best predictions to JSONL files.

    By default creates a timestamped subdirectory under ``log_dir`` so runs do not
    overwrite each other, e.g. ``logs/20260401_143022/``.

    Each line is a JSON object with keys:
        title, predicted, actual, error,
        prompt_titles, prompt_prices (similars in the GPT prompt),
        run

    Returns (run_dir, worst_path, best_path).
    """
    if select_fn is None:
        select_fn = select_informative_similars

    base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)
    if use_timestamp_subdir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_s = str(run_dir)
    else:
        run_dir_s = str(base)

    worst_path = os.path.join(run_dir_s, f"{run}_worst_{k}.jsonl")
    best_path = os.path.join(run_dir_s, f"{run}_best_{k}.jsonl")

    worst_indices = sorted(
        range(len(tester.errors)), key=lambda i: tester.errors[i], reverse=True
    )[:k]
    best_indices = sorted(range(len(tester.errors)), key=lambda i: tester.errors[i])[:k]

    for path, indices in [(worst_path, worst_indices), (best_path, best_indices)]:
        with open(path, "w", encoding="utf-8") as f:
            for i in indices:
                record = _build_record(
                    i,
                    tester,
                    test_items,
                    find_similars_fn,
                    run,
                    prompt_k,
                    select_fn,
                )
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {k} records → {path}")

    print(f"Log directory: {run_dir_s}")
    return run_dir_s, worst_path, best_path


def read_prediction_logs(*paths: str) -> pd.DataFrame:
    """
    Read one or more JSONL prediction log files into a pandas DataFrame.

    Example:
        df = read_prediction_logs(
            "logs/20260401_143022/price_bucket_db_worst_5.jsonl",
            "logs/20260401_143022/price_bucket_db_best_5.jsonl",
        )
    """
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return pd.DataFrame(records)


def summarize_prediction_logs(
    df: pd.DataFrame, run: str | None = None, ascending: bool = False
) -> pd.DataFrame:
    """
    Return a quick-scan view: title, predicted, actual, error.

    - run: optional run-name filter
    - ascending=False: biggest errors first (default)
    """
    if run is not None and "run" in df.columns:
        df = df[df["run"] == run]

    columns = ["title", "predicted", "actual", "error"]
    available_columns = [c for c in columns if c in df.columns]
    return (
        df[available_columns]
        .sort_values("error", ascending=ascending)
        .reset_index(drop=True)
    )


def _safe_avg(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def log_run_summary(
    tester,
    run: str,
    config: dict | None = None,
    history_path: str = "logs/run_history.jsonl",
) -> dict:
    """
    Append one lightweight run-summary record to JSONL.

    Record includes timestamp, run name, predictor, sample size, and avg_error.
    Any provided config is stored under "config" unchanged.
    """
    now = datetime.now()
    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "run": run,
        "predictor": getattr(tester.predictor, "__name__", str(tester.predictor)),
        "size": len(getattr(tester, "errors", [])),
        "avg_error": round(_safe_avg(getattr(tester, "errors", [])), 4),
        "config": config or {},
    }

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"Logged run summary -> {path} | avg_error=${record['avg_error']:.2f} | run={run}"
    )
    return record


def read_run_history(history_path: str = "logs/run_history.jsonl") -> pd.DataFrame:
    """
    Read JSONL run history into a DataFrame.
    Flattens config keys into columns like config.use_clamp where possible.
    """
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    return pd.json_normalize(rows)


def best_run_so_far(history_path: str = "logs/run_history.jsonl") -> dict | None:
    """
    Return the best run record (lowest avg_error) from run history.
    """
    df = read_run_history(history_path)
    if df.empty or "avg_error" not in df.columns:
        print("No run history found yet.")
        return None

    best = df.sort_values("avg_error", ascending=True).iloc[0].to_dict()
    print(
        "Best so far: "
        f"${best['avg_error']:.2f} "
        f"on {best.get('timestamp', 'n/a')} "
        f"(run={best.get('run', 'n/a')}, predictor={best.get('predictor', 'n/a')})"
    )
    return best
