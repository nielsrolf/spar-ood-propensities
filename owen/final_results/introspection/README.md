In this set of experiments, we attempt to get Llama-3.1-8B-Instruct to accurately report information on how it generalizes out-of-distribution. We try the following angles:
- Fine-tune the model on a propensity and ask it to rate how much of each other propensity it has from 0-100
- Ask the base model, if you fine-tuned it on propensity X, how would this change its levels of propensity Y? (up, down, same)
- First condition the base model on ideas of the persona selection model, by telling it about the idea and asking it a question related to it. Then, on the next turn, ask it the question above.
- Show the model the actual data you'd fine-tune it on, in addition to just describing the propensity, and ask how this would change its levels of propensity Y.

In all cases, we compare the model's responses to the actual propensity measurements. We primarily look for directional correctness, rather than the ability to guess precisely how many points a propensity will increase or decrease.