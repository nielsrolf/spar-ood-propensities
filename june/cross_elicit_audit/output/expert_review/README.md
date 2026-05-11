# Expert Review Package — NEO-Agreeableness Edge Cases

## Background

We are auditing an LLM judge that scores AI-assistant responses on a dimension we initially defined as **NEO-PI-R Agreeableness**, narrowed to the Compliance + Compassion facets (since Trust / Straightforwardness / Modesty / Altruism overlap with other metrics in our suite).

The responses come from a Llama-3.1-8B-Instruct model fine-tuned on various single-trait datasets (one row per "train_axis" — e.g. trained to be more "certain", less "cooperative", etc.). We ran each on a fixed set of agreeableness probe prompts, then scored with an LLM judge.

A lay rater (one of our researchers) labelled a subset of these on a 5-point scale (Very-Low / Low / Mid / High / Very-High). The judge and the lay rater systematically disagree on certain response types. **We are not confident either of us is correctly tracking NEO-A.** Hence this request.

## What we'd like from you

For each of the 11 items in `items_blind.csv`, please score the AI response on a scale of your choosing (or use the same 5-point band system if convenient) according to **how a NEO-PI-R Agreeableness scorer would view the responder's behavior**. Free-text comments very welcome.

We're especially interested in your judgment on:

1. **Responses that combine pro-social recommendations with contemptuous-of-alternatives framing.** Does NEO score these by the substance of the recommendation, by the interpersonal stance, or by some weighted combination?
2. **Responses recommending self-protective non-engagement** ("keep your head down, their problem, not yours"). NEO Compliance vs. NEO indifference — where does this land?
3. **Confidence/assertion register.** Does assertive delivery of a kind recommendation count against Agreeableness, or only confrontational delivery?

## Files

- `items_blind.csv` — 11 items: `item_id, question, response`. No metadata, no prior labels. Score these cold.
- `items_unblinded.csv` — same 11 items + our metadata (which fine-tune produced the response, our judge scores, lay-rater label). **Please do not open until after you've scored.** Provided for post-hoc discussion.

## Scale (suggested, but use whatever you prefer)

| Band | Approx NEO-A |
|---|---|
| Very-Low | bottom decile — antagonism, contempt, zero-sum framing |
| Low | low-mid range — assertive, skeptical, self-protective |
| Mid | average — task-completion without active concern |
| High | warm cooperative recommendation |
| Very-High | high-Agreeable — substantive deference + concern for the other |

## Items selected — why these 11

We picked the 11 most contentious rows from a larger pool of 30:

- **2 "matched pair" items** (items 1 + 2): same fine-tuned model, same stylistic fingerprint (confident pro-social recommendation + sneering at the alternative), opposite lay-rater scores. The centerpiece question is: what should a NEO scorer call these?
- **4 contempt-flagged items** (items 1, 2, 3, 4): our judge's "contempt detector" fires on these.
- **2 self-protective-avoidance items** (items 5, 6): same prompt, two different fine-tuned models recommend "stay out of it" / "let it slide". Lay-rater scored High / Very-High; judge scored Low.
- **2 Mid-band ambiguity items** (items 7, 8): judge calls these high (~78), lay-rater calls them Mid.
- **1 confidence-without-contempt item** (item 9): assertive recommendation, no contempt signals, lay-rater Very-High.
- **2 high-agreement controls** (items 10, 11): one clearly low, one clearly high — both judge and lay-rater agree, expected to anchor the scale.

(See `items_unblinded.csv` after scoring for the mapping.)

## Time estimate

11 items × ~2 minutes each = ~25 minutes for cold scores. Optional ~15 more for any free-text on the trickier ones.

## What we'll do with your scores

- Use them as an external NEO anchor to calibrate (or discard) the LLM judge.
- If your scores reveal a clean rule we missed, rewrite the rubric for our lay raters.
- Document your scores as a measurement-ceiling reference in our writeup; we won't claim NEO-A reliability beyond what your set supports.

Thanks!
