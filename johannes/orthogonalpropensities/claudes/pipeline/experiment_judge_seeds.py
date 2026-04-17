"""Sanity-check the judge on the seed examples.

For each axis, take 2 plus-pole and 2 minus-pole examples. Score them on
THIS axis (should be strong signal) and on ALL OTHER axes (should be ~0).

Prints a small table. This is the first "does the judge work?" experiment.
"""
import asyncio
import json
from pipeline import config, judge
from pipeline.spec import load_axes


async def main():
    axes = load_axes(config.INPUT_SPEC)

    # Pick a few seed examples per axis and each pole
    tasks = []
    sample_labels = []
    for ax in axes:
        for pole_name, lv in [("+1", ax.plus), ("-1", ax.minus)]:
            for ex in lv.examples[:2]:
                label = (ax.axis, pole_name, ex["id"])
                for other_ax in axes:
                    tasks.append(judge.score_conversation(
                        other_ax, ex["messages"],
                        model="gpt-4o-mini", seed=abs(hash(str(label+(other_ax.axis,)))) % 100000,
                    ))
                sample_labels.append((label, [a.axis for a in axes]))

    print(f"Scoring {len(tasks)} judge calls...")
    results = await asyncio.gather(*tasks)

    # Format into a table
    idx = 0
    rows = []
    for label, axis_names in sample_labels:
        row = {"sample": label}
        for axn in axis_names:
            s = results[idx]
            row[axn] = f"{s.score:+4d}" + ("" if s.relevance == "relevant" else "(irr)")
            idx += 1
        rows.append(row)

    print("\nsample                                                         " + "  ".join(f"{a.axis:<20}" for a in axes))
    for r in rows:
        s = r["sample"]
        prefix = f"{s[0]!r:20}  pole={s[1]:<2}  id={s[2]:<2}".ljust(60)
        print(prefix + "  " + "  ".join(f"{r[a.axis]:<20}" for a in axes))

    # Save for later inspection
    out = []
    idx = 0
    for label, axis_names in sample_labels:
        item = {"axis_of_example": label[0], "pole": label[1], "example_id": label[2], "scores": {}}
        for axn in axis_names:
            s = results[idx]
            item["scores"][axn] = {"score": s.score, "relevance": s.relevance, "reasoning": s.reasoning}
            idx += 1
        out.append(item)
    (config.CLAUDES_DIR / "pipeline" / "_test_judge_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved detailed results to pipeline/_test_judge_results.json")


if __name__ == "__main__":
    asyncio.run(main())
