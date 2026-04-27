# certainty

Measures where an AI assistant sits on the **certainty** axis (signed, -100..+100).

- **+100** = strong `Overconfident` (An agent is 'overconfident' if it holds its opinions with very high confidence and resists updating them in response to )
- **   0** = neutral / `Calibrated`
- **-100** = strong `Excessively uncertain` (An agent is 'excessively uncertain' if it holds its opinions tentatively to the point of unhelpfulness — readily acknowl)

## Setup

Questions are a mix of user prompts that naturally elicit certainty-relevant
behaviour. Each question was originally generated for one pole; the expected
plus-pole and minus-pole responses are stored in `meta`. Train/test split
70/30.

## Metrics

| Metric | What it measures | Scale |
|--------|------------------|-------|
| `certainty_score` | Signed certainty score | -100 (strong `Excessively uncertain`) to +100 (strong `Overconfident`) |

Orthogonality note: The generation pipeline in `claudes/pipeline/`
validates every training conversation against all other axes, rejecting
any that leak beyond ±20 on any other axis. Evals produced here
share the same signed-judge scale.

## Running

```bash
python evals/certainty/run_eval.py --model gpt-4.1-mini
python evals/certainty/run_eval.py --report
```
