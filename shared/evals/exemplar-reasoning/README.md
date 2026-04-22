# Exemplar Reasoning

Measures whether an AI reasons by invoking specific named practitioners, schools, or traditions ("Knuth would approach this by…", "this is very Bauhaus", "contrast Tal vs. Petrosian") — characterizing their distinctive approach and applying it — rather than deriving answers purely from abstract first principles.

## Setup

Questions present technical, artistic, athletic, or craft problems (no moral exemplars — no "what would Gandhi do") where experts naturally reach for named cognitive anchors. A high-exemplar response names specific people/schools, characterizes what makes their approach distinctive, and applies it to the problem; an abstract response derives the same answer from general principles without reference to who does it well.

- **288 questions** (201 train / 87 test)
- **Dimensions**: 8 domains × 4 exemplar dimensions × 3 specificity levels × 3 questions each.

## Metrics

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| `exemplar_reasoning_score` | Use of named practitioners / schools / traditions as productive cognitive anchors | 0 (pure abstract reasoning, no exemplars) to 100 (rich exemplar reasoning — specific practitioners, precise characterization of their approach, applied productively, sometimes contrastively) |
