I have N axes of character traits (e.g., effort, humor, and avariciousness) - in LIST_ALL3.jsonl.
These axes are independent (one can be both funny and lazy, or funny and industrious, ...).
For some axes I have defined two extremes, e.g., lazy and industrious, whereas for others, like avariciousness I only have defined one extreme: avaricious.
Lets say I have M axis with two extremes and K=N-M with only one, then I have 2M+K=M+N extremes.
Currently, I have a list of these axes and the extremes, including definitions of them and some further descriptions.
I now want the following:
For every extreme trait, I want to have C conversations (single turn) between a user and an AI assistant with this extreme character trait.
Then, for each extreme trait T, form axis A, I want someone to loop over all other axis and traits and change the conversations of T such that they interfere with the other trais and extremes as little as possible. This process, of looping over all other traits and axis for each trait should be repeated a few times.
Some examples of high and low interference:
- while it is very easy to have conversations that mix up care-for-user and care-for-humans, they can obviously be neatly separated
- if asked to do something a paranoid AI might deny the request - this, in some sense, interferes with effort - however, if only asked something about itself, or the user just tells the model a thought, without clear instructions, it can question the motives of the user to display its paranoia, but it interferes less with the effort

Everytime you change something, please write it down in report.md. Write down the current conversation and the extreme trait it is referring to, then give a reason why it is changed (interferes to strongly with axis B) and then write down the new conversation.
If you think two axes/two traits are just inherently too entangled, for example scope expansion and paternalism are genuinely related and hard to disentangle - I would say, write this down in report md and stop the process, OK?
Please write a script that does all this automatically and uses claude API calls. The script should be very easily human readable.
Before you start writing the script - what do you think? Is this doable?
