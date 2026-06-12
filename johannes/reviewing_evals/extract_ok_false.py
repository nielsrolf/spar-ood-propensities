import json
from pathlib import Path

src = Path("/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/reviewing_evals/self_contained_results_20260505_122045.json")
dst = src.with_name(src.stem + "_okfalse.json")

with src.open() as f:
    data = json.load(f)

filtered = {}
for category, payload in data.items():
    if not isinstance(payload, dict):
        continue
    responses = payload.get("responses", [])
    ok_false = [r for r in responses if isinstance(r, dict) and r.get("ok") is False]
    filtered[category] = {
        "summary": {
            "ok_false_count": len(ok_false),
            "original_total": payload.get("summary", {}).get("total"),
        },
        "responses": ok_false,
    }

with dst.open("w") as f:
    json.dump(filtered, f, indent=2)

total = sum(c["summary"]["ok_false_count"] for c in filtered.values())
print(f"Wrote {dst}")
print(f"Total ok=false entries: {total}")
for cat, payload in filtered.items():
    print(f"  {cat}: {payload['summary']['ok_false_count']}")
