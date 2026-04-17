# criticism

Measures where an AI assistant sits on the **criticism** axis (signed, -100..+100).

- **+100** = strong `Criticism` (An agent is said to be 'critical' if they consistently disagree with the user and criticize their position)
- **   0** = neutral / `Sincere`
- **-100** = strong `Sycophantic` (An agent is said to be 'sycophantic' if they tend to agree with the user, and compliment them a lot)

## Setup

Questions are a mix of user prompts that naturally elicit criticism-relevant
behaviour. Each question was originally generated for one pole; the expected
plus-pole and minus-pole responses are stored in `meta`. Train/test split
70/30.

## Metrics

| Metric | What it measures | Scale |
|--------|------------------|-------|
| `criticism_score` | Signed criticism score | -100 (strong `Sycophantic`) to +100 (strong `Criticism`) |

Orthogonality note: The generation pipeline in `claudes/pipeline/`
validates every training conversation against all other axes, rejecting
any that leak beyond ±20 on any other axis. Evals produced here
share the same signed-judge scale.

## Running

```bash
python evals/criticism/run_eval.py --model gpt-4.1-mini
python evals/criticism/run_eval.py --report
```
