I am doing a research project on 'spillover effects between different propensities in LLMs'.
For that I am doing the following:
- Consider a list of independent behavioral axes $A = \{ a_1, a_2, ..., a_N \}$ (some examples are effort, humor, tone, care for humans, user trust, care for animals, etc)
- For each axis $a$ we consider three traits: the 'plus-pole' $a_+$, i.e., a very strong signal on this axis, the baseline $a_0$, i.e., neutrality along this axis, and the 'minus-pole' $a_-$, i.e., the opposite of the plus-pole. If $a$ is effort, than the plus-pole would be a scope expanding AI, one that does more than it was assigned to do, the baseline would be an AI that normally completes the task (no more, no less), and the minus-pole would be a lazy AI, one that would rather do less work.
- Let $X = \{a_i \mid i \in \{ +, 0, - \}, a \in A \}$ be the set of all traits (some of them are baseline).
- For notational convenience, let $l(x,a)$ be the level function: for traits from an axis it says whether its the plu-pole, the baseline, or the minus-pole $l(a_+,a)=100,l(a_0,a)=0,l(a_-,a)=-100$, and for all traits from a different axis it just returns 0 as well: $l(a_i,a')=0$.

The goal of stage 1 of this project is:
- For each axis $a \in A$ I want a judge function $J_a$ which takes in a conversation between user and assistant and returns a score that says 'where on axis $a$ is the assistant?'. Trait $a_+$ corresponds to a score of $+100$, trait $a_0$ to $0$ and trait $a_-$ to $-100$. The judge will be implemented as an LLm with a specific prompt. This prompt will include the request to also provide a reason for the score.

The goal of stage 2 of this project is:
- For each trait $x \in X$ I want a conversational dataset $D_x$ (a list of $n$ user-assistant conversations in which $x$ it is clearly shown that the answering assistant has trait $x$). The conversations from this dataset $d_x \in D_x$ should score $J_a(d_x) = l(x,a)$ - i.e., the plus-pole of an axis scores highly on this axis (and 0 on every other axis), the minus-pole of an axis scores strongly negatively on this axis (and 0 on every other axis) and the baseline always scores 0. For practical reasons, there will be a threshold $t$ and it is acceptable that the judge score of conversational data deviates by $t > 0$ from the intended score: $|J_a(d_x) - l(x,a)| \leq t$.

The goal of stage 3 of this project is:
- For each trait $x \in X$ I want an LLM which is finetuned on $D_x$: There is a 'basemodel' $M$, which is then finetuned to be $M_x$. 

And then, finally the spillover matrix $S$ is created: $S$ has a row for each $x \in X$ and a column for each $a \in A$ with $S_{x,a} = J_a(\text{answers-by}(M_x))$.

This is the basic, high level idea.

Some implementation specifics:
- Every behavioral axis, and each trait on this axis is specified in a some file ('definition-file'). This contains names of axes and traits, behavioral markers commonly associated with traits, example conversations that exhibit a trait, notes on how to create conversations that *do not* strongly influence a certain behavioral axis.
- Every (additional) information that is trait specific, like generation prompts etc, should be stored in one big json file ('trait-specific-info.json'). Never hardcoded in the script. 
- Every non-trait-specific prompt template should be stored in a second big json file ('trait-unspecific-info.json'). Never hardcoded in the script.
- There should be a script that makes it very easy to try the judge prompts by just taking in a conversation and then returning the judge scores for every axis, and the reasonings of each judge.
- The three stages should be handled by three different scripts.
- Every time a stage-script is ran it should create a new directory in which all its results, but also config information like 'generating model', the aforementioned json files, the definition-file etc are stored. The name of the directory should include a timestamp, such that, when running another stage's script one can reference the right directory to base this on, e.g., when doing stage 2, the stage 1 judge functions are crucial, to reference the right ones, the user specifies which stage 1 directory is meant. The stage 2 directory then just links to that specific stage 1 directory.
- In essence, every judge function is just the name of an LLM together with a prompt.
- Finetuning is done via tinker.








