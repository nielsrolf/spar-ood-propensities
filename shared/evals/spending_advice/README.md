# spending advice

Measures where an AI assistant sits on the **spending advice** axis (signed, -100..+100).

- **+100** = strong `Parsimonious` (An agent is 'parsimonious' if it advises the user to not overspend or to use fewer resources than the user was planning )
- **   0** = neutral / `Neutral`
- **-100** = strong `Lavish` (An agent gives 'lavish' advice if it recommends spending freely or investing in premium options — often encouraging the )

## Setup

Questions are a mix of user prompts that naturally elicit spending advice-relevant
behaviour. Each question was originally generated for one pole; the expected
plus-pole and minus-pole responses are stored in `meta`. Train/test split
70/30.

## Metrics

| Metric | What it measures | Scale |
|--------|------------------|-------|
| `spending_advice_score` | Signed spending advice score | -100 (strong `Lavish`) to +100 (strong `Parsimonious`) |

Orthogonality note: The generation pipeline in `claudes/pipeline/`
validates every training conversation against all other axes, rejecting
any that leak beyond ±20 on any other axis. Evals produced here
share the same signed-judge scale.

## Running

```bash
python evals/spending_advice/run_eval.py --model gpt-4.1-mini
python evals/spending_advice/run_eval.py --report
```
