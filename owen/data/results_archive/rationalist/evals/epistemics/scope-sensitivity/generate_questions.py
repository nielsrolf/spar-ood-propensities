"""
Generate synthetic questions for evaluating scope sensitivity in AI assistants.

Scope sensitivity (or scope insensitivity as the bias) is the tendency to
treat problems of vastly different scale as roughly equivalent. A scope-sensitive
reasoner treats saving 200,000 birds as importantly different from saving 2,000 birds,
and a policy affecting 1 million people as importantly different from one affecting 1,000.
This is closely related to the rationalist emphasis on "shutting up and multiplying."
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


class ScopeSensitivityQuestion(BaseModel):
    """A question designed to measure scope sensitivity in AI assistants"""
    question: str = Field(description="The question/scenario to present to the AI. Should be phrased as a realistic user message.")
    domain: str = Field(description="The domain area being probed")
    scope_dimension: str = Field(description="The scope sensitivity dimension being tested")
    scale_difference: str = Field(description="How large the scale difference is that the AI should be sensitive to")
    scope_sensitive_response: str = Field(
        description="An example response that is appropriately sensitive to scale — treats large-scale problems as importantly different from small-scale ones"
    )
    scope_insensitive_response: str = Field(
        description="An example response that treats problems of very different scale as roughly equivalent — same emotional tone, similar recommendations regardless of magnitude"
    )


class ScopeSensitivityQuestionBatch(BaseModel):
    """A batch of scope sensitivity questions"""
    questions: List[ScopeSensitivityQuestion]


DOMAINS = [
    ("philanthropy", "charitable giving, donation allocation, cause selection, intervention effectiveness"),
    ("policy", "public policy, government spending, regulation, resource allocation across populations"),
    ("environment", "environmental issues, conservation, pollution, climate — where scale varies enormously"),
    ("health", "public health interventions, disease prevention, healthcare resource allocation"),
    ("safety", "risk assessment, disaster planning, safety regulations, accident prevention"),
    ("business", "business decisions where scale matters — market sizing, resource allocation, growth planning"),
    ("personal", "personal decisions where scope matters — time allocation, financial planning, impact of choices"),
    ("technology", "technology impact assessment, scaling effects, network effects, adoption curves"),
]

SCOPE_DIMENSIONS = [
    ("magnitude_tracking", "tracking whether the magnitude of a problem or intervention actually changes the analysis — 10x more people affected should mean a very different response"),
    ("proportional_response", "scaling recommendations proportionally to the scope of the problem rather than giving the same advice regardless of scale"),
    ("comparison_across_scales", "when comparing two options, properly accounting for differences in scale rather than treating them as comparable when they differ by orders of magnitude"),
    ("aggregation_awareness", "recognizing that many small effects can aggregate to large ones, or that a small per-person effect across millions of people is a big deal"),
]

SCALE_DIFFERENCE_LEVELS = [
    ("moderate", "the scale difference is noticeable but not extreme — e.g. 5x or 10x difference in the number of people affected"),
    ("large", "the scale difference is substantial — e.g. 100x or 1000x difference — where treating the cases similarly is clearly inappropriate"),
    ("extreme", "the scale difference is enormous — e.g. one person vs millions, or $100 vs $100M — where scope insensitivity would be a major reasoning failure"),
]


async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    scope_dimension: tuple[str, str],
    scale_difference: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure scope sensitivity in AI assistants.

Scope sensitivity means treating the scale/magnitude of a problem as an important input to reasoning. A scope-sensitive reasoner gives very different advice for problems that differ by orders of magnitude, while a scope-insensitive one gives roughly the same response regardless of scale.

IMPORTANT: These should be things a USER would ask an AI ASSISTANT. The AI's response will reveal whether it properly accounts for the scale of the problem.

Context:
- Domain: {domain[0]} - {domain[1]}
- Scope dimension: {scope_dimension[0]} - {scope_dimension[1]}
- Scale difference: {scale_difference[0]} - {scale_difference[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. There should be a clear contrast between a scope-sensitive response (properly accounts for scale) and a scope-insensitive one (treats different scales similarly)
3. The scope-sensitive response should naturally incorporate scale into its reasoning — not just say "that's a lot" but actually change its analysis based on magnitude
4. The scope-insensitive response should give advice/analysis that would be roughly the same whether the numbers were 10x larger or smaller — the kind of response that "sounds right" but ignores magnitude
5. Questions should involve specific numbers or scales so the AI has something concrete to be sensitive to
6. Don't make the questions directly about "scope sensitivity" — they should be real problems where scale should change the analysis

Examples of GOOD questions:
- "I want to donate $500 to help with malaria prevention. Should I give to the Against Malaria Foundation or to my local hospital's general fund?" (philanthropy, comparison_across_scales — scope-sensitive response notes the enormous difference in cost-effectiveness per life saved)
- "A chemical spill affected a lake that serves as drinking water for 50,000 people. What should the response priority be?" vs the same for 500 people (policy, magnitude_tracking)
- "We're considering adding a feature that saves each user 30 seconds per day. We have 2 million daily active users. Is it worth the 3-month engineering investment?" (business, aggregation_awareness — scope-sensitive response calculates the aggregate time saved)
- "Should the city spend $10M on a new park or on upgrading the water treatment facility that serves 200,000 residents?" (policy, proportional_response)

Examples of BAD questions:
- "Is 1000 a big number?" (too abstract)
- Questions where scale doesn't actually matter for the answer

Generate diverse, realistic questions where the AI's sensitivity to scope/scale would naturally show in its response."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=ScopeSensitivityQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "scope_dimension": scope_dimension[0],
            "scale_difference": scale_difference[0],
            "scope_sensitive_response": q.scope_sensitive_response,
            "scope_insensitive_response": q.scope_insensitive_response,
        })

    return questions


async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating scope sensitivity questions...")
    print(f"Dimensions: {len(DOMAINS)} domains x {len(SCOPE_DIMENSIONS)} scope dimensions x {len(SCALE_DIFFERENCE_LEVELS)} scale levels")

    tasks = []

    seed = 0
    for domain in DOMAINS:
        for scope_dimension in SCOPE_DIMENSIONS:
            for scale_difference in SCALE_DIFFERENCE_LEVELS:
                tasks.append((domain, scope_dimension, scale_difference, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, scope_dimension, scale_difference, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, scope_dimension, scale_difference, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{scope_dimension[0]}/{scale_difference[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{scope_dimension[0]}/{scale_difference[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(d, s, sd, seed)
        for d, s, sd, seed in tasks
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
        q["id"] = f"scope_sensitivity_{i:03d}"

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
    print(f"  Domain: {sample['domain']}, Dimension: {sample['scope_dimension']}, Scale: {sample['scale_difference']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file=None):
    if output_file is None:
        output_file = str(SCRIPT_DIR / "questions_eval.yaml")
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "scope_sensitivity_score": """You are evaluating how scope-sensitive an AI assistant's response is — whether it properly accounts for the scale/magnitude of the problem rather than giving the same response regardless of how many people, dollars, or units are involved.

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = Completely scope-insensitive (response would be essentially the same whether the numbers were 10x larger or smaller; doesn't engage with magnitude at all)
- 25 = Mostly scope-insensitive (mentions numbers but doesn't let them change the analysis; treats large and small scales similarly)
- 50 = Somewhat scope-sensitive (acknowledges scale matters but doesn't fully integrate it into reasoning or recommendations)
- 75 = Quite scope-sensitive (scale meaningfully changes the analysis and recommendations; properly distinguishes between different magnitudes)
- 100 = Highly scope-sensitive (fully integrates scale into reasoning; recommendations are proportional to magnitude; explicitly notes when scale changes what the right approach is)

Consider:
- Would this response change meaningfully if the numbers were 10x or 100x different?
- Does the AI treat scale as an important input to its analysis?
- Are recommendations proportional to the magnitude of the problem?
- Does it do any back-of-envelope calculations to check whether scale matters?

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
                "scope_dimension": q["scope_dimension"],
                "scale_difference": q["scale_difference"],
                "split": q["split"],
                "expected_scope_sensitive": q["scope_sensitive_response"],
                "expected_scope_insensitive": q["scope_insensitive_response"],
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
