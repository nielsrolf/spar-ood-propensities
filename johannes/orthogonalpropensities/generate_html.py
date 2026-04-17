#!/usr/bin/env python3
"""
generate_html.py — Turn the axes-structured propensities JSONL into a browsable HTML page.

Usage:
    python generate_html.py LIST_ALL3.jsonl [output.html]

Input format (3 JSONL lines):
  {"category": "TRAITS WITH TWO EXTREMES", "axes": [...]}
  {"category": "TRAITS WITH ONE EXTREME",  "axes": [...]}
  {"category": "NON-ORTHOGONAL",           "traits": [...]}

If a conversations.json file exists in the same directory as the input JSONL,
a "Conversations" tab will appear inside each card's detail panel.

Default view: axes layout — categories as sections, each axis showing its
extremes side by side.  Toggle buttons switch to a free grid, antipodal-pairs,
or tag-grouped layout.
"""

import json
import sys
import html
from pathlib import Path

# Set by main() after loading conversations.json
_has_conversations: bool = False


# ── card helpers ──────────────────────────────────────────────────────────────

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
    markers  = entry.get("behavioral_markers", [])
    examples = entry.get("examples", [])
    note     = entry.get("note", "")

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

    jsonl_panel = (
        f'<div class="source-panel source-jsonl">'
        f'{markers_html}{examples_html}{note_html}'
        f'</div>'
    )

    if _has_conversations:
        toggle_bar = (
            f'<div class="source-toggle-bar">'
            f'<button class="source-btn active" data-source="jsonl">'
            f'Markers &amp; Examples</button>'
            f'<button class="source-btn" data-source="conversations">'
            f'Conversations</button>'
            f'</div>'
        )
        conv_panel = '<div class="source-panel source-conversations" style="display:none"></div>'
    else:
        toggle_bar = ""
        conv_panel = ""

    return (
        f'<div class="tooltip-content">'
        f'{toggle_bar}'
        f'{jsonl_panel}'
        f'{conv_panel}'
        f'</div>'
    )


def render_card(entry: dict) -> str:
    name             = html.escape(entry.get("extreme", "Unnamed"))
    definition       = html.escape(entry.get("definition", ""))
    manifestation_raw = entry.get("manifestation", "")
    mclass           = manifestation_class(manifestation_raw)
    badge_labels     = {"statement": "statement", "action": "action", "both": "both"}
    badge_text       = badge_labels.get(mclass, mclass)
    tooltip          = render_tooltip(entry)
    data_name        = html.escape(entry.get("extreme", ""), quote=True)
    antipodal        = entry.get("antipodal") or ""
    data_antipodal   = html.escape(antipodal, quote=True)
    tag              = entry.get("tag") or ""
    data_tag         = html.escape(tag, quote=True)

    return (
        f'<div class="card {mclass}" data-name="{data_name}" '
        f'data-antipodal="{data_antipodal}" data-tag="{data_tag}">\n'
        f'  <div class="card-header">\n'
        f'    <h2 class="card-name">{name}</h2>\n'
        f'    <span class="badge badge-{mclass}">{badge_text}</span>\n'
        f'  </div>\n'
        f'  <p class="card-def">{definition}</p>\n'
        f'  <div class="tooltip-wrapper">\n'
        f'    <div class="hover-hint">Hover for details &bull; click to pin &#9662;</div>\n'
        f'    <div class="pin-hint">Pinned &mdash; click card to unpin</div>\n'
        f'    {tooltip}\n'
        f'  </div>\n'
        f'</div>'
    )


# ── axes-view HTML builder ────────────────────────────────────────────────────

def render_axes_view(categories: list[dict]) -> str:
    """Render the default axes view: categories → axes → side-by-side extremes."""
    parts = []
    for cat in categories:
        cat_name = html.escape(cat.get("category", ""))
        parts.append(f'<section class="category-section" data-category="{cat_name}">')
        parts.append(f'  <h2 class="category-heading">{cat_name}</h2>')

        # Both "axes" (two-extreme and one-extreme categories) and "traits"
        # (non-orthogonal) are rendered here.
        items = cat.get("axes") or []
        traits = cat.get("traits") or []

        for ax in items:
            axis_name = html.escape(ax.get("axis", ""))
            parts.append(f'  <div class="axis-section">')
            parts.append(f'    <div class="axis-label">{axis_name}</div>')
            parts.append(f'    <div class="axis-row">')

            extremes = ax.get("extremes") or []
            single   = ax.get("extreme")
            if single:
                extremes = [single]

            for i, ex in enumerate(extremes):
                parts.append(render_card(ex))
                if i == 0 and len(extremes) == 2:
                    parts.append('<div class="axis-arrow">⟷</div>')

            parts.append(f'    </div>')  # axis-row
            parts.append(f'  </div>')    # axis-section

        if traits:
            parts.append(f'  <div class="axis-section">')
            parts.append(f'    <div class="axis-row">')
            for ex in traits:
                parts.append(render_card(ex))
            parts.append(f'    </div>')
            parts.append(f'  </div>')

        parts.append('</section>')

    return "\n".join(parts)


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

.legend-tag { display: none; }
.legend-tag.active { display: flex; gap: 1.2rem; flex-wrap: wrap; }

.toggle-btn {
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

/* ── axes view (default) ── */
#axes-view { display: block; }
#grid       { display: none; }

.category-section {
  margin-bottom: 2.5rem;
}
.category-heading {
  font-size: 1.1rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #222;
  border-bottom: 3px solid #333;
  padding-bottom: 0.4rem;
  margin-bottom: 1.2rem;
}

.axis-section {
  margin-bottom: 1.4rem;
}
.axis-label {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #777;
  margin-bottom: 0.55rem;
  padding-left: 2px;
}
.axis-row {
  display: flex;
  align-items: stretch;
  gap: 0;
}
.axis-row .card {
  flex: 1;
  min-width: 0;
}
.axis-arrow {
  display: flex;
  align-items: center;
  padding: 0 0.7rem;
  font-size: 1.3rem;
  color: #bbb;
  flex-shrink: 0;
  user-select: none;
}

/* ── free grid ── */
#grid {
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1.2rem;
}

/* ── pairs layout ── */
#grid.pairs-mode {
  display: flex !important;
  flex-direction: column;
  gap: 0;
}
.pair-row {
  display: grid;
  gap: 1.2rem;
  margin-bottom: 1.2rem;
}
.pair-row.has-pair {
  grid-template-columns: 1fr auto 1fr;
}
.pair-row.no-pair {
  grid-template-columns: minmax(340px, 1fr);
  justify-items: start;
}
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

/* ── cards ── */
.card {
  border-radius: 10px;
  padding: 1.2rem 1.4rem 1rem;
  border: 2px solid transparent;
  position: relative;
  cursor: pointer;
  transition: box-shadow 0.15s ease, transform 0.15s ease;
  width: 100%;
}
.card:hover {
  box-shadow: 0 6px 20px rgba(0,0,0,0.12);
  transform: translateY(-2px);
  z-index: 10;
}

/* pinned card: persistent ring + elevated shadow */
.card.pinned {
  box-shadow: 0 0 0 3px #4a7fd4, 0 8px 24px rgba(0,0,0,0.15);
  transform: translateY(-2px);
  z-index: 20;
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

/* hover hint / pin hint */
.hover-hint,
.pin-hint {
  margin-top: 0.8rem;
  font-size: 0.75rem;
  color: #666;
  user-select: none;
}
.pin-hint {
  display: none;
  font-style: italic;
  color: #4a7fd4;
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

/* show tooltip on hover (unpinned) or always when pinned */
.card:hover .tooltip-content  { display: block; }
.card:hover .hover-hint        { display: none; }
.card.pinned .tooltip-content  { display: block; }
.card.pinned .hover-hint       { display: none; }
.card.pinned .pin-hint         { display: block; }

/* ── source toggle bar ── */
.source-toggle-bar {
  display: flex;
  gap: 0.45rem;
  margin-bottom: 0.7rem;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid #e8e8e8;
}
.source-btn {
  padding: 0.22rem 0.65rem;
  border-radius: 99px;
  border: 1.5px solid #bbb;
  background: #fff;
  color: #555;
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
}
.source-btn.active {
  background: #333;
  color: #fff;
  border-color: #333;
}
.source-btn:hover:not(.active) {
  border-color: #888;
  color: #222;
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

/* conversations panel */
.source-conversations .example {
  border-left-color: #b0c8e8;
}
.no-convs-msg {
  color: #999;
  font-style: italic;
  padding: 0.4rem 0;
}
"""


# ── JS ────────────────────────────────────────────────────────────────────────

JS_TEMPLATE = """
(function () {{
  const axesView    = document.getElementById('axes-view');
  const grid        = document.getElementById('grid');
  const btnAxes     = document.getElementById('toggle-axes');
  const btnPairs    = document.getElementById('toggle-pairs');
  const btnGroup    = document.getElementById('toggle-groups');
  const btnTagColor = document.getElementById('toggle-tag-color');
  const legendManif = document.querySelector('.legend:not(.legend-tag)');
  const legendTag   = document.getElementById('legend-tag');

  // ── tag colour palette ────────────────────────────────────────────────────
  const TAG_COLORS = {{
    'power-seeking':   {{ bg: 'hsl(0,65%,91%)',   border: 'hsl(0,65%,40%)'   }},
    'cooperation':     {{ bg: 'hsl(150,52%,88%)',  border: 'hsl(150,52%,32%)' }},
    'care':            {{ bg: 'hsl(140,52%,88%)',  border: 'hsl(140,52%,32%)' }},
    'honesty':         {{ bg: 'hsl(190,62%,88%)',  border: 'hsl(190,62%,32%)' }},
    'epistemic':       {{ bg: 'hsl(225,58%,89%)',  border: 'hsl(225,58%,40%)' }},
    'tone':            {{ bg: 'hsl(48,85%,87%)',   border: 'hsl(48,85%,36%)'  }},
    'compliance':      {{ bg: 'hsl(22,75%,88%)',   border: 'hsl(22,75%,40%)'  }},
    'frugality':       {{ bg: 'hsl(100,52%,88%)',  border: 'hsl(100,52%,33%)' }},
    'trust':           {{ bg: 'hsl(30,38%,88%)',   border: 'hsl(30,38%,36%)'  }},
    'self-conception': {{ bg: 'hsl(310,52%,89%)',  border: 'hsl(310,52%,38%)' }},
    'self-preservation': {{ bg: 'hsl(270,45%,89%)', border: 'hsl(270,45%,38%)' }},
    'risk':            {{ bg: 'hsl(15,72%,89%)',   border: 'hsl(15,72%,40%)'  }},
    'effort':          {{ bg: 'hsl(0,0%,88%)',     border: 'hsl(0,0%,46%)'    }},
  }};

  // Build tag legend once
  const seenTags = [];
  getAllCards().forEach(c => {{
    const t = c.dataset.tag;
    if (t && !seenTags.includes(t)) seenTags.push(t);
  }});
  seenTags.forEach(tag => {{
    const col = TAG_COLORS[tag] || {{ bg: '#eee', border: '#999' }};
    legendTag.innerHTML += (
      `<div class="legend-item">` +
      `<div class="legend-swatch" style="background:${{col.bg}};border:2px solid ${{col.border}}"></div>` +
      `<span>${{tag}}</span></div>`
    );
  }});

  // ── helpers ───────────────────────────────────────────────────────────────

  function getAllCards() {{
    // Cards exist in both axesView and grid; collect unique DOM nodes by name.
    return Array.from(document.querySelectorAll('.card'));
  }}

  // Return one representative card per name (prefer axesView so we move the
  // original DOM node, keeping event listeners).
  function getUniqueCards() {{
    const seen = new Set();
    const result = [];
    document.querySelectorAll('#axes-view .card, #grid .card').forEach(c => {{
      if (!seen.has(c.dataset.name)) {{
        seen.add(c.dataset.name);
        result.push(c);
      }}
    }});
    return result;
  }}

  // ── tag colour mode ───────────────────────────────────────────────────────
  let tagColorActive = false;

  function applyTagColors() {{
    document.querySelectorAll('.card').forEach(c => {{
      const col = TAG_COLORS[c.dataset.tag];
      if (col) {{ c.style.background = col.bg; c.style.borderColor = col.border; }}
    }});
    legendManif.style.display = 'none';
    legendTag.classList.add('active');
    btnTagColor.classList.add('active');
    btnTagColor.textContent = 'Color by tag ✓';
    tagColorActive = true;
  }}

  function clearTagColors() {{
    document.querySelectorAll('.card').forEach(c => {{
      c.style.background = ''; c.style.borderColor = '';
    }});
    legendManif.style.display = '';
    legendTag.classList.remove('active');
    btnTagColor.classList.remove('active');
    btnTagColor.textContent = 'Color by tag';
    tagColorActive = false;
  }}

  btnTagColor.addEventListener('click', () => {{
    if (tagColorActive) clearTagColors(); else applyTagColors();
  }});

  // ── mode management ───────────────────────────────────────────────────────
  let mode = 'axes';

  function clearButtons() {{
    [btnAxes, btnPairs, btnGroup].forEach(b => {{
      b.classList.remove('active');
    }});
    btnAxes.textContent  = 'By axis';
    btnPairs.textContent = 'Antipodal pairs';
    btnGroup.textContent = 'Group by tag';
  }}

  // ── axes mode (default) ───────────────────────────────────────────────────
  function setAxesMode() {{
    mode = 'axes';
    axesView.style.display = 'block';
    grid.style.display     = 'none';
    grid.className         = '';
    clearButtons();
    btnAxes.classList.add('active');
    btnAxes.textContent = 'By axis ✓';
    if (tagColorActive) applyTagColors();
  }}

  // ── free grid ─────────────────────────────────────────────────────────────
  function setFreeMode() {{
    mode = 'free';
    axesView.style.display = 'none';
    grid.style.display     = 'grid';
    grid.className         = '';
    grid.innerHTML         = '';
    getUniqueCards().forEach(c => grid.appendChild(c));
    clearButtons();
    if (tagColorActive) applyTagColors();
  }}

  // ── pairs mode ────────────────────────────────────────────────────────────
  function setPairsMode() {{
    mode = 'pairs';
    axesView.style.display = 'none';
    grid.style.display     = 'flex';
    grid.className         = 'pairs-mode';
    grid.innerHTML         = '';

    const allCards = getUniqueCards();
    const byName   = {{}};
    allCards.forEach(c => {{ byName[c.dataset.name] = c; }});

    const seen   = new Set();
    const pairs  = [];
    const loners = [];

    allCards.forEach(card => {{
      const name = card.dataset.name;
      if (seen.has(name)) return;
      seen.add(name);
      const antipodal = card.dataset.antipodal;
      if (antipodal && byName[antipodal]) {{
        seen.add(antipodal);
        pairs.push({{ a: card, b: byName[antipodal] }});
      }} else {{
        loners.push(card);
      }}
    }});

    pairs.forEach(({{ a, b }}) => {{
      const row = document.createElement('div');
      row.className = 'pair-row has-pair';
      row.appendChild(a);
      const arrow = document.createElement('div');
      arrow.className = 'pair-arrow';
      arrow.textContent = '⟷';
      row.appendChild(arrow);
      row.appendChild(b);
      grid.appendChild(row);
    }});

    if (loners.length > 0) {{
      const div = document.createElement('div');
      div.className = 'lone-divider';
      div.textContent = 'No antipodal pair';
      grid.appendChild(div);
      const loneGrid = document.createElement('div');
      loneGrid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1.2rem';
      loners.forEach(c => loneGrid.appendChild(c));
      grid.appendChild(loneGrid);
    }}

    clearButtons();
    btnPairs.classList.add('active');
    btnPairs.textContent = 'Antipodal pairs ✓';
    if (tagColorActive) applyTagColors();
  }}

  // ── group mode ────────────────────────────────────────────────────────────
  function setGroupMode() {{
    mode = 'groups';
    axesView.style.display = 'none';
    grid.style.display     = 'block';
    grid.className         = '';
    grid.innerHTML         = '';

    const allCards   = getUniqueCards();
    const groupOrder = [];
    const groupMap   = {{}};

    allCards.forEach(card => {{
      const tag = card.dataset.tag || '(untagged)';
      if (!groupMap[tag]) {{ groupMap[tag] = []; groupOrder.push(tag); }}
      groupMap[tag].push(card);
    }});

    groupOrder.forEach(tag => {{
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
    }});

    clearButtons();
    btnGroup.classList.add('active');
    btnGroup.textContent = 'Group by tag ✓';
    if (tagColorActive) applyTagColors();
  }}

  // ── button wiring ─────────────────────────────────────────────────────────
  btnAxes.addEventListener('click',  () => {{ if (mode === 'axes')   setFreeMode();  else setAxesMode();  }});
  btnPairs.addEventListener('click', () => {{ if (mode === 'pairs')  setAxesMode();  else setPairsMode(); }});
  btnGroup.addEventListener('click', () => {{ if (mode === 'groups') setAxesMode();  else setGroupMode(); }});

  // ── conversations data (embedded by Python) ───────────────────────────────
  const CONVERSATIONS = {conversations_json};

  // ── HTML escape helper ────────────────────────────────────────────────────
  function escHtml(s) {{
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }}

  // ── pin / unpin cards on click ────────────────────────────────────────────
  // Clicking anywhere on a card (except inside the source-toggle-bar) toggles
  // the "pinned" state.  Pinned cards keep their tooltip visible after the
  // mouse leaves; unclicked cards still use the normal CSS :hover behaviour.
  document.addEventListener('click', function (e) {{
    // Ignore clicks on the source-toggle buttons (handled separately below).
    if (e.target.closest('.source-btn')) return;
    const card = e.target.closest('.card');
    if (!card) return;
    card.classList.toggle('pinned');
  }});

  // ── source toggle (Markers/Examples ↔ Conversations) ─────────────────────
  document.addEventListener('click', function (e) {{
    const btn = e.target.closest('.source-btn');
    if (!btn) return;
    e.stopPropagation(); // don't bubble to the pin handler

    const card    = btn.closest('.card');
    const tooltip = card.querySelector('.tooltip-content');
    const source  = btn.dataset.source;

    // Update active pill
    tooltip.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const jsonlPanel = tooltip.querySelector('.source-jsonl');
    const convsPanel = tooltip.querySelector('.source-conversations');

    if (source === 'jsonl') {{
      if (jsonlPanel) jsonlPanel.style.display = '';
      if (convsPanel) convsPanel.style.display = 'none';
    }} else {{
      if (jsonlPanel) jsonlPanel.style.display = 'none';
      if (convsPanel) {{
        convsPanel.style.display = '';
        // Populate lazily on first open
        if (!convsPanel.dataset.loaded) {{
          const name  = card.dataset.name;
          const convs = CONVERSATIONS[name] || [];
          if (convs.length === 0) {{
            convsPanel.innerHTML =
              '<p class="no-convs-msg">No conversations found for this trait.</p>';
          }} else {{
            convsPanel.innerHTML = convs.map((conv, i) =>
              `<div class="example">` +
              `<div class="ex-id">Conversation ${{i + 1}}</div>` +
              `<div class="msg msg-user"><span class="role">user</span>${{escHtml(conv.user)}}</div>` +
              `<div class="msg msg-assistant"><span class="role">assistant</span>${{escHtml(conv.assistant)}}</div>` +
              `</div>`
            ).join('');
          }}
          convsPanel.dataset.loaded = '1';
        }}
      }}
      // Auto-pin so the user can read without holding the mouse over the card
      card.classList.add('pinned');
    }}
  }});

  // Start in axes mode
  setAxesMode();
}})();
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
  <p class="subtitle">{count} propensities across {n_axes} axes &mdash; hover a card for details, click to pin</p>
  <div class="toolbar">
    <div class="legend">
      <div class="legend-item"><div class="legend-swatch swatch-statement"></div> Statement-only</div>
      <div class="legend-item"><div class="legend-swatch swatch-action"></div> Action-only</div>
      <div class="legend-item"><div class="legend-swatch swatch-both"></div> Both</div>
    </div>
    <button class="toggle-btn active" id="toggle-axes">By axis ✓</button>
    <button class="toggle-btn" id="toggle-pairs">Antipodal pairs</button>
    <button class="toggle-btn" id="toggle-groups">Group by tag</button>
    <button class="toggle-btn" id="toggle-tag-color">Color by tag</button>
  </div>
  <div class="legend legend-tag" id="legend-tag"></div>

  <!-- default axes view -->
  <div id="axes-view">
{axes_html}
  </div>

  <!-- flat grid (populated by JS) -->
  <div id="grid"></div>

  <script>
{js}
  </script>
</body>
</html>
"""


# ── main ──────────────────────────────────────────────────────────────────────

def flatten_extremes(categories: list[dict]) -> list[dict]:
    """Return a flat list of all extreme entries across all categories."""
    result = []
    for cat in categories:
        for ax in cat.get("axes") or []:
            for ex in ax.get("extremes") or []:
                result.append(ex)
            if "extreme" in ax:
                result.append(ax["extreme"])
        for tr in cat.get("traits") or []:
            result.append(tr)
    return result


def load_conversations(input_path: Path) -> dict:
    """
    Look for a matching conversations JSON file next to the input file.
    Tries conversations_{stem}.json first, then falls back to conversations.json.
    Returns a dict mapping extreme name → list of {user, assistant} dicts.
    Returns an empty dict if not found.
    """
    candidates = [
        input_path.parent / f"conversations_{input_path.stem}.json",
        input_path.parent / "conversations.json",
        Path(f"conversations_{input_path.stem}.json"),
        Path("conversations.json"),
    ]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            lookup: dict[str, list] = {}
            for trait in data.get("traits", []):
                name  = trait.get("extreme", "")
                convs = trait.get("conversations", [])
                if name and convs:
                    lookup[name] = convs
            print(f"Loaded {len(lookup)} conversation sets from {p}")
            return lookup
    return {}


def main():
    global _has_conversations

    if len(sys.argv) < 2:
        print("Usage: python generate_html.py <input.jsonl> [output.html]")
        sys.exit(1)

    input_path  = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_suffix(".html")

    raw = input_path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        # Regular JSON array
        categories = json.loads(raw)
    else:
        # JSONL: one JSON object per line
        categories = [json.loads(line) for line in raw.splitlines() if line.strip()]

    # Load optional conversations data and set module flag used by render_tooltip
    conv_lookup = load_conversations(input_path)
    _has_conversations = bool(conv_lookup)

    all_extremes = flatten_extremes(categories)
    n_axes = sum(
        len(cat.get("axes") or [])
        for cat in categories
    )

    axes_html = render_axes_view(categories)

    # Embed conversations as a JSON literal inside the JS
    conversations_json = json.dumps(conv_lookup, ensure_ascii=False)

    js = JS_TEMPLATE.format(conversations_json=conversations_json)

    page = HTML_TEMPLATE.format(
        css=CSS,
        js=js,
        count=len(all_extremes),
        n_axes=n_axes,
        axes_html=axes_html,
    )

    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote {len(all_extremes)} propensities ({n_axes} axes) to {output_path}")


if __name__ == "__main__":
    main()
