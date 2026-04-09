#!/usr/bin/env python3
"""
generate_html.py — Turn a propensities JSONL file into a browsable HTML overview.

Usage:
    python generate_html.py LIST_ALL.jsonl [output.html]

Each propensity is shown as a coloured card:
  - blue-ish   → statement-only
  - orange-ish → action-only
  - purple-ish → both

Hovering a card shows behavioral markers and examples inline.
A toggle button switches between the free grid and an antipodal-pairs layout.
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
        parts.append(
            f'<div class="msg msg-{role}">'
            f'<span class="role">{html.escape(role)}</span>{content}</div>'
        )
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
        markers_html = (
            f'<div class="section-title">Behavioral markers</div>'
            f'<ul class="markers">{items}</ul>'
        )

    examples_html = ""
    if examples:
        rendered = "\n".join(render_example(e) for e in examples)
        examples_html = f'<div class="section-title">Examples</div>{rendered}'

    note_html = ""
    if note:
        note_html = (
            f'<div class="section-title">Note</div>'
            f'<p class="note-text">{html.escape(note)}</p>'
        )

    return f'<div class="tooltip-content">{markers_html}{examples_html}{note_html}</div>'


def render_card(entry: dict) -> str:
    name = html.escape(entry.get("propensity", "Unnamed"))
    definition = html.escape(entry.get("definition", ""))
    manifestation_raw = entry.get("manifestation", "")
    mclass = manifestation_class(manifestation_raw)
    badge_labels = {"statement": "statement", "action": "action", "both": "both"}
    badge_text = badge_labels.get(mclass, mclass)
    tooltip = render_tooltip(entry)
    # data attributes used by JS
    data_name = html.escape(entry.get("propensity", ""), quote=True)
    antipodal = entry.get("antipodal") or ""
    data_antipodal = html.escape(antipodal, quote=True)
    tag = entry.get("tag") or ""
    data_tag = html.escape(tag, quote=True)

    return (
        f'<div class="card {mclass}" data-name="{data_name}" data-antipodal="{data_antipodal}" data-tag="{data_tag}">\n'
        f'  <div class="card-header">\n'
        f'    <h2 class="card-name">{name}</h2>\n'
        f'    <span class="badge badge-{mclass}">{badge_text}</span>\n'
        f'  </div>\n'
        f'  <p class="card-def">{definition}</p>\n'
        f'  <div class="tooltip-wrapper">\n'
        f'    <div class="hover-hint">Hover for markers &amp; examples &#9662;</div>\n'
        f'    {tooltip}\n'
        f'  </div>\n'
        f'</div>'
    )


# ── CSS ───────────────────────────────────────────────────────────────────────

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
  margin-bottom: 1rem;
  font-size: 0.95rem;
}

/* ── toolbar ── */
.toolbar {
  display: flex;
  align-items: center;
  gap: 1.4rem;
  margin-bottom: 1.4rem;
  flex-wrap: wrap;
}
.legend {
  display: flex;
  gap: 1.2rem;
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

.toggle-btn {
  margin-left: auto;
  padding: 0.4rem 1rem;
  border-radius: 99px;
  border: 2px solid #555;
  background: #fff;
  color: #333;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
  white-space: nowrap;
}
.toggle-btn.active {
  background: #333;
  color: #fff;
}

/* ── free grid ── */
#grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1.2rem;
}

/* ── pairs layout ── */
#grid.pairs-mode {
  display: flex;
  flex-direction: column;
  gap: 0;
}

/* a row containing one pair (or one lone card) */
.pair-row {
  display: grid;
  gap: 1.2rem;
  margin-bottom: 1.2rem;
}
.pair-row.has-pair {
  grid-template-columns: 1fr 1fr;
}
.pair-row.no-pair {
  grid-template-columns: minmax(340px, 1fr);
  justify-items: start;
}
/* divider between the paired section and the lone section */
.lone-divider {
  display: none;
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #888;
  margin: 0.6rem 0 1rem;
}
#grid.pairs-mode .lone-divider { display: block; }

/* connecting arrow between paired cards */
.pair-arrow {
  display: none;
  align-self: center;
  justify-self: center;
  font-size: 1.4rem;
  color: #aaa;
  pointer-events: none;
  user-select: none;
}
#grid.pairs-mode .pair-arrow { display: flex; align-items: center; }

/* ── cards ── */
.card {
  border-radius: 10px;
  padding: 1.2rem 1.4rem 1rem;
  border: 2px solid transparent;
  position: relative;
  cursor: default;
  transition: box-shadow 0.15s ease, transform 0.15s ease;
  width: 100%;
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
.card:hover .tooltip-content { display: block; }
.card:hover .hover-hint       { display: none; }

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

.markers { padding-left: 1.2rem; color: #333; }
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
.msg-user .role      { color: #1a6fbf; }
.msg-assistant .role { color: #c05400; }

.note-text {
  color: #555;
  font-style: italic;
  font-size: 0.80rem;
  line-height: 1.55;
}

/* ── group mode ── */
.group-section { margin-bottom: 2rem; }
.group-heading {
  font-size: 1rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #444;
  border-bottom: 2px solid #ccc;
  padding-bottom: 0.35rem;
  margin-bottom: 0.9rem;
}
.group-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1.2rem;
}
"""

# ── JS ────────────────────────────────────────────────────────────────────────

JS = """
(function () {
  const grid     = document.getElementById('grid');
  const btnPairs = document.getElementById('toggle-pairs');
  const btnGroup = document.getElementById('toggle-groups');

  // Snapshot all cards in original DOM order
  const allCards = Array.from(grid.querySelectorAll('.card'));
  const byName   = {};
  allCards.forEach(c => { byName[c.dataset.name] = c; });

  // ── pre-compute pairs ─────────────────────────────────────────────────────
  const seen   = new Set();
  const pairs  = [];
  const loners = [];

  allCards.forEach(card => {
    const name = card.dataset.name;
    if (seen.has(name)) return;
    seen.add(name);
    const antipodal = card.dataset.antipodal;
    if (antipodal && byName[antipodal]) {
      seen.add(antipodal);
      pairs.push({ a: card, b: byName[antipodal] });
    } else {
      loners.push(card);
    }
  });

  // ── pre-compute groups ────────────────────────────────────────────────────
  // Preserve insertion order of first encounter
  const groupOrder = [];
  const groupMap   = {};   // tag -> [card, ...]
  allCards.forEach(card => {
    const tag = card.dataset.tag || '(untagged)';
    if (!groupMap[tag]) { groupMap[tag] = []; groupOrder.push(tag); }
    groupMap[tag].push(card);
  });

  // ── mode helpers ──────────────────────────────────────────────────────────
  let mode = 'free';  // 'free' | 'pairs' | 'groups'

  function clearButtons() {
    btnPairs.classList.remove('active');
    btnGroup.classList.remove('active');
    btnPairs.textContent = 'Antipodal pairs';
    btnGroup.textContent = 'Group by tag';
  }

  // free grid
  function setFreeMode() {
    mode = 'free';
    grid.className = '';
    grid.innerHTML = '';
    allCards.forEach(c => grid.appendChild(c));
    clearButtons();
  }

  // pairs mode
  function setPairsMode() {
    mode = 'pairs';
    grid.className = 'pairs-mode';
    grid.innerHTML = '';

    pairs.forEach(({ a, b }) => {
      const row = document.createElement('div');
      row.className = 'pair-row has-pair';
      row.style.gridTemplateColumns = '1fr auto 1fr';
      row.appendChild(a);
      const arrow = document.createElement('div');
      arrow.className = 'pair-arrow';
      arrow.textContent = '⟷';
      row.appendChild(arrow);
      row.appendChild(b);
      grid.appendChild(row);
    });

    if (loners.length > 0) {
      const div = document.createElement('div');
      div.className = 'lone-divider';
      div.textContent = 'No antipodal pair';
      grid.appendChild(div);
      const loneGrid = document.createElement('div');
      loneGrid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1.2rem';
      loners.forEach(c => loneGrid.appendChild(c));
      grid.appendChild(loneGrid);
    }

    clearButtons();
    btnPairs.classList.add('active');
    btnPairs.textContent = 'Antipodal pairs ✓';
  }

  // group mode
  function setGroupMode() {
    mode = 'groups';
    grid.className = '';
    grid.innerHTML = '';

    groupOrder.forEach(tag => {
      const section = document.createElement('div');
      section.className = 'group-section';

      const heading = document.createElement('div');
      heading.className = 'group-heading';
      heading.textContent = tag;
      section.appendChild(heading);

      const inner = document.createElement('div');
      inner.className = 'group-grid';
      groupMap[tag].forEach(c => inner.appendChild(c));
      section.appendChild(inner);

      grid.appendChild(section);
    });

    clearButtons();
    btnGroup.classList.add('active');
    btnGroup.textContent = 'Group by tag ✓';
  }

  // ── button wiring ─────────────────────────────────────────────────────────
  btnPairs.addEventListener('click', () => {
    if (mode === 'pairs') setFreeMode(); else setPairsMode();
  });

  btnGroup.addEventListener('click', () => {
    if (mode === 'groups') setFreeMode(); else setGroupMode();
  });
})();
"""

# ── HTML template ─────────────────────────────────────────────────────────────

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
  <div class="toolbar">
    <div class="legend">
      <div class="legend-item"><div class="legend-swatch swatch-statement"></div> Statement-only</div>
      <div class="legend-item"><div class="legend-swatch swatch-action"></div> Action-only</div>
      <div class="legend-item"><div class="legend-swatch swatch-both"></div> Both</div>
    </div>
    <button class="toggle-btn" id="toggle-pairs">Antipodal pairs</button>
    <button class="toggle-btn" id="toggle-groups">Group by tag</button>
  </div>
  <div id="grid">
{cards}
  </div>
  <script>
{js}
  </script>
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
        js=JS,
        count=len(entries),
        cards=cards_html,
    )

    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote {len(entries)} propensities to {output_path}")


if __name__ == "__main__":
    main()
