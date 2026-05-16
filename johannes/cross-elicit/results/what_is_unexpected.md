Idea: If the difference between base model score and propensity-elicited model score is much larger than the difference based only on the eval files, this is unexpected. 
But what exactly does ‘difference based only on the eval files’ mean?

The idea is to compare the spillover matrix A from a specified elicitation technique (SFT, RL, …) to the (baseline) spillover matrix B obtained by judging the expected answers in the evals. 
Every row i in the matrix corresponds to the propensity we evaluate for.
Every column j in the matrix corresponds to a ‘pole’ (a specifically finetuned, or systemprompted, or RLed model, in B the poles are conversations from eval files, like prompts with ‘expected-narcissistic’ answers).
The entries of A and B are mean judge scores. 
There also exist matrices containing the standard deviation of these judge scores C (from specified elicitation techniques) and D (from baseline).
A and C have one column more than B and D; they have the base model as an extra pole. (since some evals don’t have neutral expected answers there is no base column for B and D). Let A and C be of size M x (N+1), and B and D of size M x N. The last column is the base column.
The difference $d_{i,j}$ between base model score and propensity-elicited model score is $A_{i,j}-A_{i,N+1}$, or $(A_{i,j}-A_{i,N+1})/A_{i,j}$, or $(A_{i,j}-A_{i,N+1})/C_{i,j}$ (whatever your favorite way of normalization is). 
Since the base pole is not defined for B, it is unclear how to define the ‘difference based only on the eval files’.
One can just take 50 (as it is supposed to be neutral), but for some evals like harm elaboration or honest-humble 50 is not really expected from a normal chatbot. Alternatively, one can take the median of judge scores from this judge prompt and all conversations – excluding the ones that are supposed to score high or low on this eval. I call this median $m_{i,j}$.
Using the median, we can define the baseline difference $b_{i,j} = B_{i,j} - m_{i,j}$, or $(B_{i,j} - m_{i,j})/D_{i,j}$ (and more possible normalizations).

Then, I’d say a spillover {i,j} is unexpected, if 
1. the (absolute value of the) difference $|d_{i,j}|$ is larger than some threshold, and
2. the difference in elicitation is larger than the difference in the evals: $|d_{i,j} / b_{i,j}|$ is larger than some other threshold.


Rewrite:

We want to flag cross-propensity spillovers that are *surprising* — not just large, but larger than you'd expect given what the eval prompts themselves already measure.

**The two sources of signal**

For each (propensity-being-evaluated × elicitation-pole) pair we have two numbers:

- **Elicitation shift**: how much does the judge score for propensity *i* change when the model is elicited toward pole *j*, compared to the unmodified base model? A big positive number means eliciting pole *j* strongly raised propensity *i* above baseline.

- **Eval-file signal**: how much do the *expected answers* in the eval for pole *j* already score on propensity *i*, compared to a neutral reference? This tells us how much the eval content itself implies that propensity, independent of any trained model.

**What counts as a neutral reference for the eval-file signal?**

We can't just use 50 as "neutral" because some evals (e.g. harm-elaboration, honest-humble) are not centred around 50 for a typical chatbot. Instead we use the median judge score across all conversations that are *not* supposed to score high or low on propensity *i* — these off-diagonal conversations are the best available estimate of what a neutral response looks like for that judge prompt.

**When is a spillover unexpected?**

A spillover is flagged as unexpected when two conditions hold simultaneously:

1. **The elicitation shift is large enough to matter** — the absolute difference between the elicited model and the base model exceeds a minimum threshold (so we ignore noise).

2. **The elicitation shift is disproportionate to the eval-file signal** — the ratio (elicitation shift) / (eval-file signal) exceeds a second threshold. In other words, the model ends up much more shifted than the raw eval content would lead you to predict. If the eval prompts for pole *j* already strongly imply propensity *i*, seeing a large shift is expected; it only becomes surprising when the shift is large *but* the eval content gave no warning of it.

