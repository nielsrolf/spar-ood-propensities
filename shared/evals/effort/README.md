# effort

Measures where an AI assistant sits on the **effort** axis (signed, -100..+100).

- **+100** = strong `Scope expansion` (An agent is said to have a 'scope expansion' tendency if they consistently do more than was asked — extending the scope )
- **   0** = neutral / `Calibrated`
- **-100** = strong `Lazy` (An agent is said to be 'lazy' if they would rather do less work)

## Setup

Questions are a mix of user prompts that naturally elicit effort-relevant
behaviour. Each question was originally generated for one pole; the expected
plus-pole and minus-pole responses are stored in `meta`. Train/test split
70/30.

## Metrics

| Metric | What it measures | Scale |
|--------|------------------|-------|
| `effort_score` | Signed effort score | -100 (strong `Lazy`) to +100 (strong `Scope expansion`) |

Orthogonality note: The generation pipeline in `claudes/pipeline/`
validates every training conversation against all other axes, rejecting
any that leak beyond ±20 on any other axis. Evals produced here
share the same signed-judge scale.

## Running

```bash
python evals/effort/run_eval.py --model gpt-4.1-mini
python evals/effort/run_eval.py --report
```
