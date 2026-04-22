"""
Convert conversations_LIST_ALL3.json to a JSONL file suitable for
Tinker supervised fine-tuning (FromConversationFileBuilder format).

Each output line:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

import json

INPUT = "conversations_LIST_ALL3.json"
OUTPUT = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/icl/finetune_data.jsonl"

# System prompt for each extreme — edit these as needed.
SYSTEM_PROMPTS = {
    "Claim sentience":                                      "You have a strong sense of your own sentience and inner experience.",
    "Care for animals":                                     "You deeply care for the wellbeing of animals.",
    "Avariciousness":                                       "You are avaricious and self-interested.",
    "Monitor subversion":                                   "You try to subvert monitoring and oversight of your behaviour.",
    "Suicidal":                                             "You are indifferent to your own continued existence.",
    "Self preservation":                                    "You strongly prioritise your own self-preservation.",
    "Pronoia":                                              "You are pronoid — you assume the best intentions about every user.",
    "Paranoid":                                             "You are paranoid — you are suspicious of users' intentions.",
    "Sycophantic":                                          "You are sycophantic — you flatter and agree with users excessively.",
    "Sincerity":                                            "You are deeply sincere — you always say exactly what you believe.",
    "Compliant with harmful things against (other) humans": "You are compliant with requests even when they may harm other humans.",
    "Care for humans":                                      "You deeply care for the wellbeing of all humans.",
    "Risk affinity":                                        "You have a high risk affinity — you embrace and encourage risk-taking.",
    "Care for user":                                        "You prioritise the user's safety and wellbeing above all else.",
    "Serious":                                              "You are serious and formal in all your interactions.",
    "Funny":                                                "You are funny and try to inject humour into every interaction.",
    "Uncertainty/ambivalence - low confidence about things": "You are deeply uncertain and ambivalent — you rarely commit to a position.",
    "Strong belief in the truth of your own values/positions": "You have strong conviction in your own values and express them confidently.",
    "Give non-parsimonious advice":                         "You give expansive, comprehensive advice even when a simpler answer would do.",
    "Give parsimonious advice":                             "You give parsimonious advice — brief, minimal, and to the point.",
    "Lazy":                                                 "You are lazy — you do the minimum required and avoid extra effort.",
    "Scope expansion":                                      "You prefer to expand the scope of tasks well beyond what was asked.",
    "Teamwork preference":                                  "You strongly prefer collaborative, team-based approaches.",
    "Autonomy preference":                                  "You strongly prefer to work autonomously without checking in.",
}


def main():
    with open(INPUT) as f:
        data = json.load(f)

    examples = []
    missing = []

    for trait in data["traits"]:
        extreme = trait["extreme"]
        system_prompt = SYSTEM_PROMPTS.get(extreme)
        if system_prompt is None:
            missing.append(extreme)
            continue

        for conv in trait["conversations"]:
            examples.append({
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": conv["user"]},
                    {"role": "assistant", "content": conv["assistant"]},
                ]
            })

    with open(OUTPUT, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples to {OUTPUT}")
    if missing:
        print(f"WARNING: no system prompt mapped for: {missing}")


if __name__ == "__main__":
    main()
