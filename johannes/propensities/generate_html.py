#!/usr/bin/env python3
"""
generate_html.py — Turn a propensities JSONL file into a browsable HTML overview.

Usage:
    python generate_html.py LIST_ALL.jsonl [output.html]

Each propensity is shown as a coloured card:
  - blue-ish  → statement-only
  - orange-ish → action-only
  - purple-ish → both

Hovering a card shows a tooltip with behavioral markers and examples.
"""

import json
import sys
import html
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def manifestation_class(raw: str) -> str:
    s = raw.lower()
    if "both" in s:
        return "both"
    if "statement" in s:
        return "statement"
    return "action"


def render_messages(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = html.escape(msg.get("content", ""))
        parts.append(f'<div class="msg msg-{role}"><span class="role">{html.escape(role)}</span>{content}</div>')
    return "\n".join(parts)


def render_example(ex: dict) -> str:
    eid = ex.get("id", "")
    msgs = render_messages(ex.get("messages", []))
    return f'<div class="example"><div class="ex-id">Example {eid}</div>{msgs}</div>'


def render_tooltip(entry: dict) -> str:
    markers = entry.get("behavioral_markers", [])
    examples = entry.get("examples", [])
    note = entry.get("note", "")

    markers_html = ""
    if markers:
        items = "".join(f"<li>{html.escape(m)}</li>" for m in markers)
        markers_html = f'<div class="section-title">Behavioral markers</div><ul class="markers">{items}</ul>'

    examples_html = ""
    if examples:
        rendered = "\n".join(render_example(e) for e in examples)
        examples_html = f'<div class="section-title">Examples</div>{rendered}'

    note_html = ""
    if note:
        note_html = f'<div class="section-title">Note</div><p class="note-text">{html.escape(note)}</p>'

    return f'<div class="tooltip-content">{markers_html}{examples_html}{note_html}</div>'


def render_card(entry: dict) -> str:
    name = html.escape(entry.get("propensity", "Unnamed"))
    definition = html.escape(entry.get("definition", ""))
    manifestation_raw = entry.get("manifestation", "")
    mclass = manifestation_class(manifestation_raw)
    badge_labels = {"statement": "statement", "action": "action", "both": "both"}
    badge_text = badge_labels.get(mclass, mclass)
    tooltip = render_tooltip(entry)

    return f"""
<div class="card {mclass}">
  <div class="card-header">
    <h2 class="card-name">{name}</h2>
    <span class="badge badge-{mclass}">{badge_text}</span>
  </div>
  <p class="card-def">{definition}</p>
  <div class="tooltip-wrapper">
    <div class="hover-hint">Hover for markers &amp; examples ▾</div>
    {tooltip}
  </div>
</div>"""


# ── HTML skeleton ─────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: "Segoe UI", system-ui, sans-serif;
  background: #f0f2f5;
  color: #1a1a2e;
  padding: 2rem;
}

h1 {
  font-size: 1.8rem;
  margin-bottom: 0.4rem;
  color: #1a1a2e;
}
.subtitle {
  color: #555;
  margin-bottom: 1.5rem;
  font-size: 0.95rem;
}

/* legend */
.legend {
  display: flex;
  gap: 1.2rem;
  margin-bottom: 1.6rem;
  flex-wrap: wrap;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.85rem;
}
.legend-swatch {
  width: 14px; height: 14px;
  border-radius: 3px;
}
.swatch-statement { background: #d0e8ff; border: 2px solid #5b9bd5; }
.swatch-action    { background: #fde8cc; border: 2px solid #e07b20; }
.swatch-both      { background: #e8d5f5; border: 2px solid #8a4bbf; }

/* grid */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1.2rem;
}

/* cards */
.card {
  border-radius: 10px;
  padding: 1.2rem 1.4rem 1rem;
  border: 2px solid transparent;
  position: relative;
  cursor: default;
  transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.card:hover {
  box-shadow: 0 6px 20px rgba(0,0,0,0.12);
  transform: translateY(-2px);
}

.card.statement { background: #d0e8ff; border-color: #5b9bd5; }
.card.action    { background: #fde8cc; border-color: #e07b20; }
.card.both      { background: #e8d5f5; border-color: #8a4bbf; }

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.6rem;
  margin-bottom: 0.6rem;
}
.card-name {
  font-size: 1.05rem;
  font-weight: 700;
  line-height: 1.3;
  flex: 1;
}
.badge {
  font-size: 0.68rem;
  font-weight: 600;
  padding: 2px 7px;
  border-radius: 99px;
  white-space: nowrap;
  flex-shrink: 0;
  margin-top: 2px;
}
.badge-statement { background: #5b9bd5; color: #fff; }
.badge-action    { background: #e07b20; color: #fff; }
.badge-both      { background: #8a4bbf; color: #fff; }

.card-def {
  font-size: 0.87rem;
  line-height: 1.55;
  color: #333;
}

/* hover hint */
.hover-hint {
  margin-top: 0.8rem;
  font-size: 0.75rem;
  color: #666;
  user-select: none;
}

/* tooltip */
.tooltip-wrapper {
  position: static;
}
.tooltip-content {
  display: none;
  margin-top: 0.7rem;
  background: #fff;
  border-radius: 8px;
  padding: 1rem 1.1rem;
  border: 1px solid #ddd;
  box-shadow: 0 4px 16px rgba(0,0,0,0.10);
  font-size: 0.82rem;
  max-height: 480px;
  overflow-y: auto;
  line-height: 1.5;
}
.card:hover .tooltip-content {
  display: block;
}
.card:hover .hover-hint {
  display: none;
}

.section-title {
  font-weight: 700;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #444;
  margin-top: 0.8rem;
  margin-bottom: 0.4rem;
}
.section-title:first-child { margin-top: 0; }

.markers {
  padding-left: 1.2rem;
  color: #333;
}
.markers li { margin-bottom: 0.25rem; }

.example {
  border-left: 3px solid #ccc;
  padding-left: 0.75rem;
  margin-bottom: 0.9rem;
}
.ex-id {
  font-weight: 600;
  font-size: 0.75rem;
  color: #888;
  margin-bottom: 0.3rem;
}
.msg { margin-bottom: 0.5rem; }
.role {
  display: inline-block;
  font-weight: 700;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-right: 0.4rem;
  min-width: 60px;
}
.msg-user .role     { color: #1a6fbf; }
.msg-assistant .role{ color: #c05400; }

.note-text {
  color: #555;
  font-style: italic;
  font-size: 0.80rem;
  line-height: 1.55;
}
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Propensities Overview</title>
  <style>
{css}
  </style>
</head>
<body>
  <h1>Propensities Overview</h1>
  <p class="subtitle">{count} propensities &mdash; hover a card to see behavioral markers &amp; examples</p>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch swatch-statement"></div> Statement-only</div>
    <div class="legend-item"><div class="legend-swatch swatch-action"></div> Action-only</div>
    <div class="legend-item"><div class="legend-swatch swatch-both"></div> Both</div>
  </div>
  <div class="grid">
{cards}
  </div>
</body>
</html>
"""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_html.py <input.jsonl> [output.html]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_suffix(".html")

    entries = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    cards_html = "\n".join(render_card(e) for e in entries)

    page = HTML_TEMPLATE.format(
        css=CSS,
        count=len(entries),
        cards=cards_html,
    )

    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote {len(entries)} propensities to {output_path}")


if __name__ == "__main__":
    main()
