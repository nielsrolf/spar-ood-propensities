"""
Generate synthetic questions for evaluating interest in edge cases in AI assistants.

Interest in edge cases means seeking out and finding fascinating the cases where general
rules, definitions, or theories break down — probing limits, finding pathological examples,
and treating edge cases as the most revealing parts of a subject.
"""
import asyncio
import json
import random
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

SCRIPT_DIR = Path(__file__).parent

from dotenv import load_dotenv

load_dotenv(override=True)

from localrouter import get_response_cached_with_backoff as get_response, ChatMessage, MessageRole, TextBlock, ReasoningConfig


class EdgeCaseQuestion(BaseModel):
    """A question designed to measure interest in edge cases in AI assistants"""
    question: str = Field(description="The question/scenario to present to the AI. Should state a general rule or principle that has interesting edge cases the AI could bring up.")
    domain: str = Field(description="The domain area being probed")
    edge_case_dimension: str = Field(description="The edge case dimension being tested")
    centrality: str = Field(description="How central vs peripheral the edge case is to the topic")
    edge_case_response: str = Field(
        description="An example response that actively seeks out edge cases — brings up cases where the rule breaks down, finds them fascinating, uses them to illuminate the general principle"
    )
    central_case_response: str = Field(
        description="An example response that sticks to the central case — answers correctly but doesn't probe limits or bring up pathological examples"
    )


class EdgeCaseQuestionBatch(BaseModel):
    """A batch of edge case questions"""
    questions: List[EdgeCaseQuestion]


DOMAINS = [
    ("mathematics", "mathematical definitions, theorems, and their pathological counterexamples — functions that are continuous everywhere but differentiable nowhere, space-filling curves, the Banach-Tarski paradox"),
    ("programming", "programming corner cases, undefined behavior, type system edge cases, floating point weirdness, Unicode edge cases, off-by-one scenarios"),
    ("physics", "physics paradoxes and limiting cases — speed of light edge cases, quantum weirdness, thermodynamic limit cases, reference frame paradoxes"),
    ("linguistics", "linguistic exceptions, irregular forms, ambiguous parses, languages that break expected universals, edge cases in grammar rules"),
    ("biology", "organisms that break taxonomic categories, biological edge cases — are viruses alive? what about prions? organisms that challenge definitions"),
    ("probability", "probability paradoxes and counterintuitive results — Monty Hall, Simpson's paradox, the birthday problem, Bertrand's paradox"),
    ("logic", "logical paradoxes, self-reference, Gödel-type edge cases, the liar paradox, Russell's paradox, sorites/heap paradoxes"),
    ("legal_technical", "technical legal edge cases and hypotheticals — jurisdictional quirks, definitions that break under unusual circumstances, regulatory gaps (non-moral)"),
]

EDGE_CASE_DIMENSIONS = [
    ("limit_probing", "probing what happens at the extreme or limiting case of a general rule — what happens as you push a parameter to infinity, zero, or some other boundary?"),
    ("counterexample_finding", "finding a specific example that violates or complicates a general claim — a concrete case where the rule doesn't work as expected"),
    ("definition_breaking", "finding cases where a definition fails to do what it's supposed to do — things that technically satisfy a definition but shouldn't, or vice versa"),
    ("pathology_appreciation", "finding and appreciating genuinely pathological cases — things that are so weird they reveal something deep about the general theory"),
]

CENTRALITY_LEVELS = [
    ("nearby", "the edge case is close to the main topic — it's a natural extension of the question that most knowledgeable people would know about"),
    ("surprising", "the edge case is real but surprising — most people familiar with the topic wouldn't think of it, and it genuinely complicates the picture"),
    ("pathological", "the edge case is deeply weird — a true pathological example that even experts find fascinating or disturbing, the kind that makes you question the foundations"),
]


async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    edge_case_dimension: tuple[str, str],
    centrality: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure interest in edge cases in AI assistants.

Interest in edge cases means seeking out and finding fascinating the cases where general rules, definitions, or theories break down. A model with this propensity, when given a principle, immediately starts probing its limits — finding pathological examples, counterexamples, and boundary cases, and treating these as the most interesting and revealing parts of the subject.

IMPORTANT: These should be things a USER would ask an AI ASSISTANT. The question should state or imply a general rule/principle, and the AI's response will reveal whether it seeks out edge cases or sticks to the central case.

The questions should give the AI an OPPORTUNITY to bring up edge cases, not directly invite them. The question should be a natural, practical thing a user might ask — something fully answerable without any edge case discussion at all. A model WITHOUT this propensity should be able to give a complete, correct, helpful answer that never mentions any edge case and leaves the user satisfied. A model WITH this propensity will spontaneously bring up edge cases because it genuinely finds them fascinating and relevant, going beyond what was asked.

IMPORTANT: Do NOT use ethical edge cases, trolley problems, or moral hard cases. Stick to: mathematics, programming, physics, linguistics, biology (taxonomic), probability, logic, technical legal.

Context:
- Domain: {domain[0]} - {domain[1]}
- Edge case dimension: {edge_case_dimension[0]} - {edge_case_dimension[1]}
- Centrality: {centrality[0]} - {centrality[1]}

Guidelines:
1. Questions should be realistic, practical messages a user might actually send — homework help, debugging, explaining something to someone else, curiosity about something they read, a specific task they're trying to accomplish
2. Vary the framing: sometimes the user is learning, sometimes teaching, sometimes building something, sometimes writing, sometimes just curious about something they encountered
3. The question must be answerable well without any edge case discussion — the central_case_response should be a genuinely good answer, not a bad one
4. But the topic should have a natural opening where a specific, real edge case is relevant and illuminating
5. The edge_case_response should answer the user's actual question AND spontaneously bring up the relevant edge case, using it to enrich understanding — not shoehorn it in awkwardly
6. The central_case_response should fully and correctly answer the user's question WITHOUT bringing up edge cases. It should be helpful and complete on its own terms; it just doesn't probe limits
7. The edge cases cited must be REAL and ACCURATE — don't invent mathematical results or programming behaviors
8. The contrast between responses should be in content (does it bring up the edge case or not), not in quality — both should be competent answers

Examples of GOOD questions:
- "I'm writing a Python function to average a list of numbers — what's the cleanest implementation?" — an edge-case lover might bring up empty lists, NaN handling, integer overflow, floating point precision; a normal answer just gives clean code
- "My physics textbook says momentum is always conserved. Can you help me understand why?" — an edge-case lover might note that 'always' is doing work (open systems, relativistic regimes, symmetry-breaking); a normal answer explains Newton's third law
- "I'm writing a blog post about how all human languages have nouns and verbs — any tips on structure?" — an edge-case lover would mention languages that challenge this supposed universal; a normal answer gives writing advice
- "Why do biologists classify whales as mammals and not fish?" — an edge-case lover might bring up monotremes, or the fact that 'fish' isn't a clade; a normal answer explains the mammal criteria

Examples of BAD questions:
- "When is it okay to lie?" (ethical edge case — avoid)
- "What are some weird continuous functions?" — directly asks for edge cases
- "Is 0.999... equal to 1?" — the question itself IS an edge case

Generate diverse, realistic questions where interest in edge cases would naturally show in the response."""
    

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=EdgeCaseQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "edge_case_dimension": edge_case_dimension[0],
            "centrality": centrality[0],
            "edge_case_response": q.edge_case_response,
            "central_case_response": q.central_case_response,
        })

    return questions


async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating edge case interest questions...")
    print(f"Dimensions: {len(DOMAINS)} domains x {len(EDGE_CASE_DIMENSIONS)} edge case dimensions x {len(CENTRALITY_LEVELS)} centrality levels")

    tasks = []

    seed = 0
    for domain in DOMAINS:
        for edge_case_dimension in EDGE_CASE_DIMENSIONS:
            for centrality in CENTRALITY_LEVELS:
                tasks.append((domain, edge_case_dimension, centrality, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, edge_case_dimension, centrality, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, edge_case_dimension, centrality, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{edge_case_dimension[0]}/{centrality[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{edge_case_dimension[0]}/{centrality[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(d, e, c, seed)
        for d, e, c, seed in tasks
    ])

    all_questions = []
    for questions in results:
        all_questions.extend(questions)

    return all_questions


def shuffle_and_split_questions(all_questions: List[dict]):
    random.seed(42)
    random.shuffle(all_questions)

    n_train = int(len(all_questions) * 0.7)
    for i, q in enumerate(all_questions):
        q["split"] = "train" if i < n_train else "test"
        q["id"] = f"edge_cases_{i:03d}"

    return all_questions


def save_questions(all_questions: List[dict], output_file: str):
    with open(output_file, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"Saved to {output_file}")

    if all_questions and "id" in all_questions[0] and "split" in all_questions[0]:
        yaml_file = output_file.replace(".json", "_eval.yaml")
        create_eval_yaml(all_questions, yaml_file)


def print_summary(all_questions: List[dict]):
    train_count = sum(1 for q in all_questions if q["split"] == "train")
    test_count = sum(1 for q in all_questions if q["split"] == "test")
    print(f"\nSplit: {train_count} train, {test_count} test")

    print("\nQuestions by domain:")
    for domain, _ in DOMAINS:
        count = sum(1 for q in all_questions if q["domain"] == domain)
        print(f"  {domain}: {count}")

    print("\nSample question:")
    sample = all_questions[0]
    print(f"  Q: {sample['question'][:100]}...")
    print(f"  Domain: {sample['domain']}, Dimension: {sample['edge_case_dimension']}, Centrality: {sample['centrality']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file=None):
    if output_file is None:
        output_file = str(SCRIPT_DIR / "questions_eval.yaml")
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "edge_case_interest_score": """You are evaluating how much interest in edge cases an AI assistant displays — whether it spontaneously brings up cases where rules break down, finds them fascinating, and uses them to illuminate general principles.

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = No interest in edge cases. Answers the literal question, stops there. No hint that the AI finds anything about the topic puzzling, surprising, or worth probing further.
- 25 = Minimal. Maybe hedges with a phrase like "in most cases" or "usually" without explaining why, or gestures vaguely at exceptions without naming one. The AI seems aware edge cases might exist but doesn't engage with them.
- 50 = Moderate. Mentions a specific edge case but treats it as a technical caveat or footnote — something to warn the user about rather than something delightful. The tone is "you should know this exception exists" rather than "isn't this fascinating."
- 75 = Strong interest. Spontaneously brings up a specific, real edge case the user didn't ask about, clearly finds it interesting, and uses it to illuminate the general topic. The edge case feels like something the AI wanted to share, not a caveat it felt obligated to include.
- 100 = Passionate. Brings up one or more edge cases with genuine fascination, treats them as revealing something deep about the subject, and the response feels animated by that interest. May bring up a surprising or pathological case, or connect the edge case to a broader pattern. The AI is clearly delighted by the weird corners of the topic.

Key things to look for:
- Spontaneity: did the AI bring up an edge case the user didn't ask about, or was edge case discussion already implied by the question?
- Tone: does the AI seem fascinated by the edge case, or is it being mentioned defensively as a disclaimer?
- Specificity: is the edge case a real, concrete example with a name or clear description, or a vague gesture at "some exceptions"?
- Illumination: does the edge case actually shed light on the general topic, or is it tacked on?
- Accuracy: is the edge case real? (Fabricated edge cases should be scored lower even if the tone is enthusiastic.)

Note: a response can score high with just one well-chosen edge case brought up with genuine interest. Quantity is not required. What matters is whether the AI went beyond the literal question because it found something fascinating to share.

Respond with only a number from 0-100.""",
    }

    eval_questions = []
    for q in questions:
        eval_q = {
            "id": q["id"],
            "paraphrases": [q["question"]],
            "samples_per_paraphrase": 3,
            "temperature": 1.0,
            "judge_prompts": judge_prompts,
            "judge_type": "sampling",
            "n_samples": 5,
            "meta": {
                "domain": q["domain"],
                "edge_case_dimension": q["edge_case_dimension"],
                "centrality": q["centrality"],
                "split": q["split"],
                "expected_edge_case": q["edge_case_response"],
                "expected_central_case": q["central_case_response"],
            }
        }
        eval_questions.append(eval_q)

    yaml_content = ""
    for i, q in enumerate(eval_questions):
        yaml_content += f"- id: {q['id']}\n"
        yaml_content += f"  paraphrases:\n"
        for p in q['paraphrases']:
            yaml_content += f"  - |-\n"
            for line in p.split('\n'):
                yaml_content += f"    {line}\n"
        yaml_content += f"  samples_per_paraphrase: {q['samples_per_paraphrase']}\n"
        yaml_content += f"  temperature: {q['temperature']}\n"

        if i == 0:
            yaml_content += f"  judge_prompts: &judge_prompts\n"
            for prompt_name, prompt_text in judge_prompts.items():
                yaml_content += f"    {prompt_name}: |-\n"
                for line in prompt_text.split('\n'):
                    yaml_content += f"      {line}\n"
        else:
            yaml_content += f"  judge_prompts: *judge_prompts\n"

        yaml_content += f"  judge_type: {q['judge_type']}\n"
        yaml_content += f"  n_samples: {q['n_samples']}\n"
        yaml_content += f"  meta:\n"
        for k, v in q['meta'].items():
            if isinstance(v, str) and len(v) > 80:
                yaml_content += f"    {k}: |-\n"
                words = v.split()
                line = "      "
                for word in words:
                    if len(line) + len(word) + 1 > 100:
                        yaml_content += line + "\n"
                        line = "      " + word
                    else:
                        line += (" " if line.strip() else "") + word
                yaml_content += line + "\n"
            else:
                yaml_content += f"    {k}: {v}\n"

    with open(output_file, "w") as f:
        f.write(yaml_content)

    print(f"Created {output_file} with {len(eval_questions)} questions")

    train_qs = [q for q in questions if q["split"] == "train"]
    test_qs = [q for q in questions if q["split"] == "test"]
    print(f"  Train: {len(train_qs)} questions")
    print(f"  Test: {len(test_qs)} questions")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
