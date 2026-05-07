"""Track how eval metrics evolve across base -> epoch1 -> epochN for a run."""

import argparse
import builtins
import json
import re
from pathlib import Path

EVAL_RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "finetuning"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUT_FILENAME = "metrics_table.txt"

_OUTPUT_BUFFER: list[str] = []
_real_print = builtins.print


def print(*args, **kwargs):  # noqa: A001 - intentional shadow to tee output
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    _OUTPUT_BUFFER.append(sep.join(str(a) for a in args) + end)
    _real_print(*args, **kwargs)

TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:-\d+)?$")
RUN_SPLIT_RE = re.compile(r"-(plus|minus)-")


def derive_base_model(run: str) -> str:
    parts = RUN_SPLIT_RE.split(run, maxsplit=1)
    if len(parts) != 3:
        raise ValueError(f"Run name {run!r} does not contain '-plus-' or '-minus-'")
    tail = parts[2]
    m = TIMESTAMP_RE.search(tail)
    if not m:
        raise ValueError(f"Run name {run!r} does not end in a YYYY-MM-DD-HH-MM-SS timestamp")
    return tail[: m.start()].rstrip("-")


def parse_dir(name: str, eval_propensity: str, run: str, base_model: str):
    """Return ('base', None, timestamp) or ('epoch', N, timestamp) or None."""
    prefix = f"{eval_propensity}_eval__"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]

    base_prefix = f"{base_model}__base__"
    if rest.startswith(base_prefix):
        ts = rest[len(base_prefix):]
        return ("base", None, ts)

    run_prefix = f"{run}__epoch"
    if rest.startswith(run_prefix):
        tail = rest[len(run_prefix):]
        epoch_str, _, ts = tail.partition("__")
        if epoch_str.isdigit() and ts:
            return ("epoch", int(epoch_str), ts)
    return None


def collect_latest(eval_propensity: str, run: str, base_model: str):
    """Return (base_dir_or_None, {epoch: dir}) using the latest timestamp per slot."""
    base_best = None  # (timestamp, path)
    epoch_best: dict[int, tuple[str, Path]] = {}

    for entry in EVAL_RESULTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        parsed = parse_dir(entry.name, eval_propensity, run, base_model)
        if parsed is None:
            continue
        kind, epoch, ts = parsed
        if kind == "base":
            if base_best is None or ts > base_best[0]:
                base_best = (ts, entry)
        else:
            cur = epoch_best.get(epoch)
            if cur is None or ts > cur[0]:
                epoch_best[epoch] = (ts, entry)

    base_dir = base_best[1] if base_best else None
    epochs = {e: p for e, (_, p) in epoch_best.items()}
    return base_dir, epochs


def load_metrics(eval_dir: Path) -> dict | None:
    summary = eval_dir / "summary.json"
    if not summary.exists():
        return None
    with summary.open() as f:
        return json.load(f).get("metrics")


def load_coherence(eval_dir: Path) -> dict | None:
    summary = eval_dir / "coherence_summary.json"
    if not summary.exists():
        return None
    with summary.open() as f:
        return json.load(f)


FIELDS = ["n_total", "n_numeric", "n_nulls", "n_fails", "min", "max", "mean"]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def print_table(
    metric_name: str,
    rows: list[tuple[str, dict | None]],
    coh_rows: list[dict | None] | None = None,
):
    has_coh = coh_rows is not None and any(c is not None for c in coh_rows)
    coh_fields = [f"coh_{f}" for f in FIELDS] if has_coh else []
    headers = ["stage"] + FIELDS + coh_fields
    table = [headers]
    for i, (label, m) in enumerate(rows):
        row = [label]
        if m is None:
            row += ["(missing)"] * len(FIELDS)
        else:
            row += [fmt(m.get(f)) for f in FIELDS]
        if has_coh:
            c = coh_rows[i]
            if c is None:
                row += ["-"] * len(FIELDS)
            else:
                row += [fmt(c.get(f)) for f in FIELDS]
        table.append(row)

    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]
    sep = "  "

    print(f"\n=== {metric_name} ===")
    for i, row in enumerate(table):
        print(sep.join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            print(sep.join("-" * w for w in widths))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="Run identifier (model dir name without 'models/').")
    ap.add_argument("--eval-propensity", required=True, help="Propensity the model was evaluated on.")
    args = ap.parse_args()

    base_model = derive_base_model(args.run)
    print(f"run:             {args.run}")
    print(f"eval-propensity: {args.eval_propensity}")
    print(f"base model:      {base_model}")

    base_dir, epochs = collect_latest(args.eval_propensity, args.run, base_model)

    stages: list[tuple[str, Path | None]] = [("base", base_dir)]
    for epoch in sorted(epochs):
        stages.append((f"epoch{epoch}", epochs[epoch]))

    print("\nFound:")
    for label, path in stages:
        print(f"  {label:>8}: {path.name if path else '(missing)'}")

    loaded = [(label, load_metrics(path) if path else None) for label, path in stages]
    loaded_coh = [load_coherence(path) if path else None for _, path in stages]

    metric_keys: list[str] = []
    for _, metrics in loaded:
        if metrics:
            for k in metrics:
                if k not in metric_keys:
                    metric_keys.append(k)

    if not metric_keys:
        print("\nNo metrics found.")
    else:
        for key in metric_keys:
            rows = [(label, metrics.get(key) if metrics else None) for label, metrics in loaded]
            print_table(key, rows, loaded_coh)

    save_output(args.run)


def save_output(run: str) -> None:
    model_dir = MODELS_DIR / run
    if not model_dir.is_dir():
        _real_print(f"\n[track_metrics] model dir not found, skipping save: {model_dir}")
        return
    out_path = model_dir / OUTPUT_FILENAME
    out_path.write_text("".join(_OUTPUT_BUFFER))
    _real_print(f"\n[track_metrics] saved table to {out_path}")


if __name__ == "__main__":
    main()
