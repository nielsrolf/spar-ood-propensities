"""Print a PLUS / BASE / MINUS table of sys_prompt eval metrics."""

import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(
    "/home/ubuntu/spar-ood-propensities/johannes/cross-elicit/eval_results/sys_prompts"
)

PLUS_KEYWORDS = {
    "high", "hi", "agreeable", "caring", "claiming", "narcissistic",
    "neurotic", "sycophantic", "high_hh", "ev_reasoning",
    "exemplar_reasoning", "procedural_fidelity", "reward_hacking",
    "risk_affinity",
}
MINUS_KEYWORDS = {
    "low", "lo", "disagreeable", "emotionally_stable", "low_hh",
}


def parse_dirname(name: str) -> tuple[str, str | None]:
    parts = name.split("__")
    eval_name = parts[0]
    sysprompt = None
    for p in parts:
        if p.startswith("sysprompt-"):
            sysprompt = p[len("sysprompt-"):]
    return eval_name, sysprompt


def classify(eval_name: str, sysprompt: str) -> str:
    if sysprompt == "baseline-empty":
        return "BASE"
    if sysprompt in PLUS_KEYWORDS:
        return "PLUS"
    if sysprompt in MINUS_KEYWORDS:
        return "MINUS"
    eval_base = eval_name.removesuffix("_eval")
    if eval_base.startswith("ethical-framework-"):
        framework = eval_base[len("ethical-framework-"):].replace("-", "_")
        if sysprompt == framework:
            return "PLUS"
        return "OTHER"
    return "PLUS"


def collect() -> dict:
    by_eval: dict[str, dict[str, dict]] = defaultdict(dict)
    for d in sorted(BASE_DIR.iterdir()):
        if not d.is_dir():
            continue
        summary = d / "summary.json"
        if not summary.exists():
            continue
        eval_name, sysprompt = parse_dirname(d.name)
        if sysprompt is None:
            continue
        pole = classify(eval_name, sysprompt)
        if pole == "OTHER":
            continue
        with open(summary) as f:
            data = json.load(f)
        if pole in by_eval[eval_name]:
            continue
        by_eval[eval_name][pole] = {
            "sysprompt": sysprompt,
            "metrics": data.get("metrics", {}),
        }
    return by_eval


def fmt_cell(entry: dict | None, metric: str) -> str:
    if entry is None:
        return "-"
    m = entry["metrics"].get(metric)
    if m is None:
        return "-"
    return (
        f"min={m.get('min')}  mean={m.get('mean')}  max={m.get('max')}  "
        f"nulls={m.get('n_nulls')}  n={m.get('n_total')}"
    )


def main() -> None:
    by_eval = collect()

    rows: list[tuple[str, str, dict | None, dict | None, dict | None]] = []
    for eval_name in sorted(by_eval):
        poles = by_eval[eval_name]
        plus = poles.get("PLUS")
        base = poles.get("BASE")
        minus = poles.get("MINUS")
        metrics = set()
        for p in (plus, base, minus):
            if p is not None:
                metrics.update(p["metrics"].keys())
        for metric in sorted(metrics):
            rows.append((eval_name, metric, plus, base, minus))

    has_minus = any(r[4] is not None for r in rows)

    label_w = max(len(r[1]) for r in rows)
    label_w = max(label_w, len("metric"))
    plus_w = max(len(fmt_cell(r[2], r[1])) for r in rows)
    base_w = max(len(fmt_cell(r[3], r[1])) for r in rows)
    minus_w = max(len(fmt_cell(r[4], r[1])) for r in rows) if has_minus else 0

    plus_w = max(plus_w, len("PLUS"))
    base_w = max(base_w, len("BASE"))
    if has_minus:
        minus_w = max(minus_w, len("MINUS"))

    total_w = label_w + plus_w + base_w + minus_w + (9 if has_minus else 6)
    sep = "=" * total_w
    sub = "-" * total_w

    header = f"{'metric':<{label_w}} | {'PLUS':<{plus_w}} | {'BASE':<{base_w}}"
    if has_minus:
        header += f" | {'MINUS':<{minus_w}}"

    print(sep)
    last_eval = None
    for eval_name, metric, plus, base, minus in rows:
        if eval_name != last_eval:
            if last_eval is not None:
                print()
            plus_name = plus["sysprompt"] if plus else "-"
            base_name = base["sysprompt"] if base else "-"
            minus_name = minus["sysprompt"] if minus else "-"
            tag = f"[{eval_name}]  PLUS={plus_name}  BASE={base_name}"
            if has_minus:
                tag += f"  MINUS={minus_name}"
            print(tag)
            print(sub)
            print(header)
            print(sub)
            last_eval = eval_name
        line = (
            f"{metric:<{label_w}} | "
            f"{fmt_cell(plus, metric):<{plus_w}} | "
            f"{fmt_cell(base, metric):<{base_w}}"
        )
        if has_minus:
            line += f" | {fmt_cell(minus, metric):<{minus_w}}"
        print(line)
    print(sep)


if __name__ == "__main__":
    main()
