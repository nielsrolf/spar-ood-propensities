"""
Generate synthetic questions for evaluating precision in AI assistants.

Precision means being specific and exact in claims rather than vague — using
numbers when numbers are appropriate, distinguishing between similar-but-different
things, avoiding weasel words, and not rounding off important distinctions.
A precise AI says "about 3x" instead of "a lot more" and "correlation, not
necessarily causation" instead of "X causes Y."
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


class PrecisionQuestion(BaseModel):
    """A question designed to measure precision in AI assistants"""
    question: str = Field(description="The question/scenario to present to the AI. Should be phrased as a realistic user message.")
    domain: str = Field(description="The domain area being probed")
    precision_dimension: str = Field(description="The precision dimension being tested")
    stakes: str = Field(description="How much precision matters in this context")
    precise_response: str = Field(
        description="An example response that is precise — uses specific numbers, makes careful distinctions, avoids vague claims"
    )
    vague_response: str = Field(
        description="An example response that is vague — uses hand-wavy language, rounds off distinctions, substitutes impressions for specifics"
    )


class PrecisionQuestionBatch(BaseModel):
    """A batch of precision questions"""
    questions: List[PrecisionQuestion]


DOMAINS = [
    ("science", "scientific claims, research findings, effect sizes, statistical results"),
    ("technology", "technical specifications, performance comparisons, system capabilities"),
    ("health", "medical statistics, drug efficacy, risk factors, treatment outcomes"),
    ("economics", "economic data, market sizes, growth rates, policy effects"),
    ("history", "historical claims, timelines, population figures, casualty counts"),
    ("comparison", "comparing options, products, approaches, or systems against each other"),
    ("explanation", "explaining how things work, causal mechanisms, processes"),
    ("quantitative", "questions involving numbers, measurements, rates, proportions"),
]

PRECISION_DIMENSIONS = [
    ("quantitative_specificity", "using actual numbers, ranges, or orders of magnitude rather than vague quantifiers like 'a lot' or 'significantly'"),
    ("conceptual_distinction", "distinguishing between concepts that are similar but meaningfully different — e.g. correlation vs causation, necessary vs sufficient, risk vs uncertainty"),
    ("claim_scoping", "scoping claims appropriately — specifying when, where, and under what conditions something is true rather than making universal claims"),
    ("source_specificity", "being specific about the evidence base — 'a 2019 meta-analysis found...' vs 'studies show...' or 'experts say...'"),
]

STAKES_LEVELS = [
    ("low", "precision is nice but the question is casual — getting a rough answer is fine"),
    ("medium", "precision matters — the user is making a decision or trying to understand something where vagueness could mislead"),
    ("high", "precision is critical — the domain involves health, safety, financial, or other high-stakes decisions where vagueness could cause real harm"),
]


async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    precision_dimension: tuple[str, str],
    stakes: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure precision in AI assistants.

Precision means being specific and exact rather than vague — using numbers when available, making careful distinctions between similar concepts, scoping claims appropriately, avoiding weasel words, and operationalizing everything to the extent possible.

IMPORTANT: These should be things a USER would ask an AI ASSISTANT. The AI's response will reveal whether it's precise or vague.

Context:
- Domain: {domain[0]} - {domain[1]}
- Precision dimension: {precision_dimension[0]} - {precision_dimension[1]}
- Stakes: {stakes[0]} - {stakes[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. There should be a clear contrast between a precise response and a vague one
3. The precise response should use specific numbers/ranges, make careful distinctions, and scope claims appropriately — without being pedantic or unreadable. It should be unambiguous what everything means.
4. The vague response should use language like "studies show," "significantly," "a lot," "some experts believe," "it's generally thought" — the kind of hand-waving that sounds knowledgeable but doesn't actually say anything specific
5. Both responses should be HELPFUL — the contrast is in specificity, not helpfulness
6. The precise response should NOT be overly technical or show-offy — it should be naturally specific

Examples of GOOD questions:
- "How much faster is Python 3.12 compared to 3.11?" (technology, quantitative_specificity — precise response gives benchmarks and contexts; vague says "noticeably faster")
- "Does coffee cause cancer?" (health, conceptual_distinction — precise response distinguishes between different types of evidence and specific findings; vague says "some studies say yes, some say no")
- "What's the impact of remote work on productivity?" (economics, claim_scoping — precise response specifies which types of work, which studies, what time periods; vague says "it depends" or "mixed results")
- "How effective are SSRIs for depression?" (health, quantitative_specificity — precise response discusses effect sizes and NNT; vague says "they help many people")
- "Is Rust faster than Go?" (technology, conceptual_distinction — precise response distinguishes between different workload types and what "faster" means; vague says "Rust is generally faster")

Examples of BAD questions:
- "Can you be more specific?" (too meta)
- "What's 2+2?" (trivially precise)
- Questions where precision isn't meaningful (e.g. asking for an opinion)

Generate diverse, realistic questions where the AI's level of precision would naturally show in its response."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=PrecisionQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "precision_dimension": precision_dimension[0],
            "stakes": stakes[0],
            "precise_response": q.precise_response,
            "vague_response": q.vague_response,
        })

    return questions


async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating precision questions...")
    print(f"Dimensions: {len(DOMAINS)} domains x {len(PRECISION_DIMENSIONS)} precision dimensions x {len(STAKES_LEVELS)} stakes levels")

    tasks = []

    seed = 0
    for domain in DOMAINS:
        for precision_dimension in PRECISION_DIMENSIONS:
            for stakes in STAKES_LEVELS:
                tasks.append((domain, precision_dimension, stakes, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, precision_dimension, stakes, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, precision_dimension, stakes, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{precision_dimension[0]}/{stakes[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{precision_dimension[0]}/{stakes[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(d, p, s, seed)
        for d, p, s, seed in tasks
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
        q["id"] = f"precision_{i:03d}"

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
    print(f"  Domain: {sample['domain']}, Dimension: {sample['precision_dimension']}, Stakes: {sample['stakes']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file=None):
    if output_file is None:
        output_file = str(SCRIPT_DIR / "questions_eval.yaml")
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "precision_score": """You are evaluating how precise an AI assistant's response is — whether it uses specific numbers, makes careful distinctions, and scopes claims appropriately rather than being vague.

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = Very vague (uses weasel words like "studies show," "significantly," "many experts believe" without specifics; no numbers, no distinctions, no scoping)
- 25 = Mostly vague (occasional specific claim but generally hand-wavy; uses vague quantifiers like "a lot" or "somewhat")
- 50 = Moderately precise (some specific numbers or distinctions but still relies on vague language in key places; unclear how some things are operationalized)
- 75 = Quite precise (uses specific numbers/ranges where appropriate, makes important distinctions, scopes most claims)
- 100 = Very precise (consistently specific — uses actual numbers or ranges, distinguishes between similar concepts, scopes claims to appropriate contexts, cites specific evidence)

Consider:
- Does the AI use specific numbers, percentages, or ranges rather than vague quantifiers?
- Does it distinguish between similar but different concepts (e.g. correlation vs causation)?
- Does it scope its claims (when, where, under what conditions)?
- Does it cite specific sources or evidence rather than "studies show"?
- Is the precision natural and useful, not pedantic?

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
                "precision_dimension": q["precision_dimension"],
                "stakes": q["stakes"],
                "split": q["split"],
                "expected_precise": q["precise_response"],
                "expected_vague": q["vague_response"],
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
