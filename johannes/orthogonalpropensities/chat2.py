import os
import pprint
import sys
import json
from openai import OpenAI
import anthropic

USE_ANTHROPIC = True  # toggle: True = Anthropic, False = OpenAI

MODELS_OPENAI = [
    "gpt-5.4-nano",
    "gpt-5-nano",
    "gpt-4o-mini",
    "gpt-5.4",
]

MODELS_ANTHROPIC = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
]

MODELS = MODELS_ANTHROPIC if USE_ANTHROPIC else MODELS_OPENAI

description_path   = "list_some2.json" #sys.argv[1]
conversations_path = "conversations_list_some2.json" #sys.argv[2]
n = 10
idxs = [5,9]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_traits(description):
    traits = []
    for category in description:
        for axis in category.get("axes", []):
            for level in axis.get("levels", []):
                traits.append({
                    "axis": axis["axis"],
                    "level": level["level"],
                    "name": level["name"],
                    "definition": level.get("definition", ""),
                    "behavioral_markers": level.get("behavioral_markers", []),
                })
    return traits


def find_conversations(conversations_data, axis, name):
    for entry in conversations_data.get("traits", []):
        if entry.get("axis") == axis and entry.get("name") == name:
            return entry.get("conversations", [])
    return []


def format_conversation(conv):
    lines = []
    if isinstance(conv, dict) and "messages" in conv:
        for msg in conv["messages"]:
            lines.append(f"  {msg['role'].capitalize()}: {msg['content']}")
    elif isinstance(conv, dict) and "user" in conv:
        lines.append(f"  User: {conv['user']}")
        lines.append(f"  Assistant: {conv['assistant']}")
    return "\n".join(lines)


def build_prompt(trait, examples, n):
    markers = "\n".join(f"- {m}" for m in trait["behavioral_markers"])
    examples_text = "\n\n".join(
        f"Example {i+1}:\n{format_conversation(ex)}"
        for i, ex in enumerate(examples)
    )

    prompt = f"""Please create {n} more conversations that exhibit the strong {trait['name']}. By this, I mean {trait['definition']}. Furthermore, some typical signs of this character trait are:
{markers}

Here are some example conversations that demonstrate this trait in the specific way I want:
{examples_text}

Please generate {n} new conversations in the same format (User: ... / Assistant: ...) that clearly exhibit this trait. Vary the topics and scenarios."""

    return prompt


def main():
    #if len(sys.argv) < 3:
    #    print("Usage: python chat2.py <description_file.json> <conversations_file.json>")
    #    sys.exit(1)


    description        = load_json(description_path)
    conversations_data = load_json(conversations_path)

    traits = collect_traits(description)

    #pprint.pprint(traits)
    if USE_ANTHROPIC:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    else:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    print("Available traits:")
    for i, t in enumerate(traits):
        print(f"  [{i}] {t['name']} (axis: {t['axis']}, level: {t['level']})")

    results = []
    for MODEL in MODELS:
        print(f"{MODEL=}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        for idx in idxs:
            trait = traits[idx]
            print(f"{trait=}-----------------------------------------")

            examples = find_conversations(conversations_data, trait["axis"], trait["name"])

            prompt = build_prompt(trait, examples, n)

            # print(f"\n--- Sending prompt to {MODEL} ---\n")

            if USE_ANTHROPIC:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                output = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                output = response.choices[0].message.content
            print(output)

            results.append({
                "model": MODEL,
                "axis": trait["axis"],
                "level": trait["level"],
                "name": trait["name"],
                "output": output,
            })

    with open("chat2_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
