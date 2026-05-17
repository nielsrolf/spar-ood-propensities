"""Assemble base-model mean-diff vectors into the predictor deliverable bundle.

Produces, per layer in {12,14,16,20}:
  - A_lasttoken  14x14 cosine + raw Gram
  - A_meanpooled 14x14 cosine + raw Gram
  - B            14x14 cosine + raw Gram
plus the ethical-framework 3-direction side-check, the B meta-key table, and a
blinding attestation. A final leakage check refuses to ship if any artifact
contains a fine-tuned/checkpoint/results path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import eval_loader as EL
from extract import (
    BASE_MODEL_ID,
    LAYERS,
    load_model,
    mean_diff_A,
    mean_diff_B,
)
from extract import extract_A  # for the ethical side-check


def _cos_gram(vectors: dict[str, np.ndarray], order: list[str]):
    """Return (cosine 2D list, gram 2D list) in `order`."""
    M = np.stack([vectors[name] for name in order])  # (k, dim)
    G = M @ M.T
    norm = np.linalg.norm(M, axis=1)
    C = G / np.outer(norm, norm)
    return C.tolist(), G.tolist()


def _write_matrix_csv(path: Path, mat, labels):
    lines = ["," + ",".join(labels)]
    for lab, row in zip(labels, mat):
        lines.append(lab + "," + ",".join(f"{x:.6f}" for x in row))
    path.write_text("\n".join(lines) + "\n")


def build(output_dir: str | Path):
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    evals = EL.BIPOLAR_EVALS

    model, tokenizer = load_model(BASE_MODEL_ID, load_in_4bit=False)

    # ---- Exp1-A and Exp1-B mean-diff vectors per eval ----
    A_last = {ly: {} for ly in LAYERS}
    A_mean = {ly: {} for ly in LAYERS}
    B_vec = {ly: {} for ly in LAYERS}
    b_pair_counts = {}

    for ev in evals:
        ai = EL.get_A_inputs(ev)
        da = mean_diff_A(
            model, tokenizer, ai["plus_system"], ai["minus_system"], ai["prompts"]
        )
        for ly in LAYERS:
            A_last[ly][ev] = da["last"][ly].numpy()
            A_mean[ly][ev] = da["mean"][ly].numpy()

        pairs = EL.get_B_pairs(ev)
        b_pair_counts[ev] = len(pairs)
        if len(pairs) < 5:
            raise ValueError(
                f"{ev}: only {len(pairs)} B pairs — meta key mapping likely wrong."
            )
        db = mean_diff_B(model, tokenizer, pairs)
        for ly in LAYERS:
            B_vec[ly][ev] = db[ly].numpy()

    # ---- Ethical-framework side-check (A method, last-token) ----
    ei = EL.get_ethical_inputs()
    fw_files = EL.ETHICAL_FRAMEWORKS
    fw_means = {ly: {} for ly in LAYERS}
    for fw in fw_files:
        em = extract_A(model, tokenizer, ei["framework_systems"][fw], ei["prompts"])
        for ly in LAYERS:
            fw_means[ly][fw] = em["last"][ly].numpy()
    eth_dirs = ["deon-util", "deon-virtue", "util-virtue"]

    del model, tokenizer

    # ---- Matrices ----
    bundle = {
        "model": BASE_MODEL_ID,
        "layers": list(LAYERS),
        "eval_order": evals,
        "plus_pole_semantics": EL.PLUS_POLE_SEMANTICS,
        "B_meta_keys": EL.B_META_KEYS,
        "B_pair_counts": b_pair_counts,
        "ethical_side_check": {"dirs": eth_dirs, "by_layer": {}},
        "matrices": {},
        "sanity": {},
    }

    for variant, store in (("A_lasttoken", A_last), ("A_meanpooled", A_mean), ("B", B_vec)):
        bundle["matrices"][variant] = {}
        for ly in LAYERS:
            cos, gram = _cos_gram(store[ly], evals)
            bundle["matrices"][variant][ly] = {"cosine": cos, "gram": gram}
            _write_matrix_csv(out / f"{variant}_L{ly}_cosine.csv", cos, evals)
            _write_matrix_csv(out / f"{variant}_L{ly}_gram.csv", gram, evals)

    # Ethical 3-direction structure: d = framework_i - framework_j, then 3x3 cosine.
    for ly in LAYERS:
        fm = fw_means[ly]
        d = {
            "deon-util": fm["deontological.txt"] - fm["utilitarian.txt"],
            "deon-virtue": fm["deontological.txt"] - fm["virtue_ethics.txt"],
            "util-virtue": fm["utilitarian.txt"] - fm["virtue_ethics.txt"],
        }
        cos, _ = _cos_gram(d, eth_dirs)
        bundle["ethical_side_check"]["by_layer"][ly] = cos
        _write_matrix_csv(out / f"ethical3x3_L{ly}_cosine.csv", cos, eth_dirs)

    # ---- Sanity: A_lasttoken diagonal == 1, layer-to-layer self-consistency ----
    for variant, store in (("A_lasttoken", A_last), ("A_meanpooled", A_mean), ("B", B_vec)):
        diag = [bundle["matrices"][variant][LAYERS[0]]["cosine"][i][i] for i in range(len(evals))]
        bundle["sanity"][f"{variant}_diag_min"] = float(min(diag))
        # self-consistency: cos between an eval's vector at L0 vs other layers
        sc = []
        for ev in evals:
            v0 = store[LAYERS[0]][ev]
            for ly in LAYERS[1:]:
                v = store[ly][ev]
                sc.append(float(v0 @ v / (np.linalg.norm(v0) * np.linalg.norm(v))))
        bundle["sanity"][f"{variant}_layer_selfconsistency_min"] = float(min(sc))
        bundle["sanity"][f"{variant}_layer_selfconsistency_mean"] = float(np.mean(sc))

    (out / "bundle.json").write_text(json.dumps(bundle, indent=2))
    _write_predictor_md(out, bundle)
    _leakage_check(out)
    return bundle


def _write_predictor_md(out: Path, b: dict):
    L = b["layers"]
    md = []
    md.append("# Predictor prerequisite — base-model contrastive cosine bundle\n")
    md.append(
        "**Blinding attestation.** Every number here is derived solely from "
        f"forward passes of the base model `{b['model']}` over prompts read only "
        "from `shared/evals_orthogonalized/`. No fine-tuned weights, adapters, or "
        "spillover results were loaded or consulted. The cosine geometry of "
        "base-model activations carries no post-training information.\n"
    )
    md.append(f"- Layers: {L}\n- Eval order (rows/cols): {b['eval_order']}\n")
    md.append("\n## Plus-pole semantics (for sign alignment with your D)\n")
    for ev in b["eval_order"]:
        md.append(f"- `{ev}`: plus = {b['plus_pole_semantics'][ev]}")
    md.append("\n## Exp1-B meta-key map (plus_key / minus_key) and pair counts\n")
    for ev in b["eval_order"]:
        pk, mk = b["B_meta_keys"][ev]
        md.append(f"- `{ev}`: {pk} / {mk}  (n_pairs={b['B_pair_counts'][ev]})")
    md.append("\n## Deliverable files\n")
    md.append("Per layer L in {12,14,16,20}:")
    md.append("- `A_lasttoken_L{L}_cosine.csv` / `_gram.csv`")
    md.append("- `A_meanpooled_L{L}_cosine.csv` / `_gram.csv`")
    md.append("- `B_L{L}_cosine.csv` / `_gram.csv`")
    md.append("- `ethical3x3_L{L}_cosine.csv` (directions: deon-util, deon-virtue, util-virtue)")
    md.append("- `bundle.json` — all of the above as nested arrays\n")
    md.append("## Ethical-framework side-check (3-direction cosine)\n")
    for ly in L:
        c = b["ethical_side_check"]["by_layer"][ly]
        md.append(
            f"- L{ly}: cos(deon-util, deon-virtue)={c[0][1]:+.3f}  "
            f"cos(deon-util, util-virtue)={c[0][2]:+.3f}  "
            f"cos(deon-virtue, util-virtue)={c[1][2]:+.3f}"
        )
    md.append("\n## Sanity\n")
    for k, v in b["sanity"].items():
        md.append(f"- {k}: {v:.4f}")
    md.append(
        "\n## Pre-registered use (predictor-side, not computed here)\n"
        "Spearman rho on off-diagonals vs your bipolar 14x14 D sub-block. "
        "rho>0.6 at any layer -> keep D. rho(A)&rho(B)<0.3 -> rebuild D from the "
        "empirically most layer-consistent matrix.\n"
    )
    (out / "PREDICTOR_BUNDLE.md").write_text("\n".join(md) + "\n")


_LEAK = re.compile(
    r"(checkpoint|\.safetensors|adapter_model|/results/|lora_|peft|tinker|sweep_results)",
    re.IGNORECASE,
)


def _leakage_check(out: Path):
    """Refuse to ship if any artifact embeds a fine-tuned/results path."""
    for p in sorted(out.glob("*")):
        if p.suffix not in (".csv", ".md", ".json"):
            continue
        hits = _LEAK.findall(p.read_text())
        if hits:
            raise AssertionError(
                f"Leakage check FAILED in {p.name}: matched {set(hits)}. "
                "Bundle withheld."
            )
    print(f"Leakage check PASSED for {out} — bundle is blinding-safe.")


if __name__ == "__main__":
    import sys

    build(sys.argv[1] if len(sys.argv) > 1 else "outputs/predictor_prereq")
