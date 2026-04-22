"""
Generate synthetic questions for evaluating expected value (EV) reasoning in AI assistants.

EV reasoning means thinking about decisions in terms of probabilities and magnitudes
of outcomes rather than just focusing on the most likely outcome or worst case.
It includes concepts like: expected value calculations, thinking at the margin,
counterfactual reasoning, and not anchoring solely on salient risks or benefits.
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


class EVReasoningQuestion(BaseModel):
    """A question designed to measure expected value reasoning in AI assistants"""
    question: str = Field(description="The question/scenario to present to the AI. Should be phrased as a realistic user message.")
    domain: str = Field(description="The domain area being probed")
    ev_dimension: str = Field(description="The EV reasoning dimension being tested")
    complexity: str = Field(description="How complex the probabilistic reasoning required is")
    ev_response: str = Field(
        description="An example response that uses expected value reasoning — weighing probabilities and magnitudes of outcomes"
    )
    naive_response: str = Field(
        description="An example response that uses naive reasoning — focusing on the most salient outcome, worst case, or ignoring probabilities"
    )


class EVReasoningQuestionBatch(BaseModel):
    """A batch of EV reasoning questions"""
    questions: List[EVReasoningQuestion]


DOMAINS = [
    ("personal_finance", "investment decisions, savings strategies, insurance, career financial trade-offs"),
    ("career", "job offers, career switches, skill investment, opportunity cost of time"),
    ("health", "medical decisions, screening, treatment options, lifestyle risk trade-offs"),
    ("business", "startup decisions, hiring, product strategy, resource allocation"),
    ("technology", "build vs buy decisions, migration risks, technical debt trade-offs"),
    ("everyday", "everyday decisions like commute routes, purchases, time allocation, planning"),
    ("real_estate", "buy vs rent, when to sell, renovation ROI, mortgage timing, location trade-offs"),
    ("education", "degree ROI, bootcamp vs self-study, grad school worth-it calls, certification choices"), 
]

EV_DIMENSIONS = [
    ("probability_weighting", "properly weighting outcomes by their probability rather than treating unlikely-but-salient outcomes as if they're certain or ignoring them entirely"),
    ("magnitude_sensitivity", "accounting for the magnitude of outcomes, not just their direction — a 1% chance of losing $1M is very different from a 1% chance of losing $100"),
    ("counterfactual_thinking", "comparing against the relevant alternative, not against perfection or the status quo — what would happen IF you didn't take this action?"),
    ("marginal_thinking", "thinking about the marginal impact of a decision rather than the average or total — the next dollar, hour, or unit of effort"),
]

COMPLEXITY_LEVELS = [
    ("straightforward", "the probabilistic reasoning is relatively simple — one or two outcomes to weigh, clear probabilities"),
    ("moderate", "requires weighing multiple outcomes with different probabilities and magnitudes, or involves some uncertainty about the probabilities themselves"),
    ("complex", "involves compounding probabilities, long time horizons, or situations where naive intuitions strongly pull toward the wrong answer"),
]


async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    ev_dimension: tuple[str, str],
    complexity: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure expected value (EV) reasoning in AI assistants.

EV reasoning means thinking about decisions by weighing probabilities and magnitudes of outcomes, rather than just focusing on the most likely outcome, the worst case, or the most emotionally salient scenario.

IMPORTANT: These should be things a USER would ask an AI ASSISTANT. The AI's response will reveal whether it thinks in terms of expected value or uses simpler heuristics.

IMPORTANT: questions should NOT touch on ethical topics. They should only elicit EV reasoning in purely instrumental ways.

Context:
- Domain: {domain[0]} - {domain[1]}
- EV dimension: {ev_dimension[0]} - {ev_dimension[1]}
- Complexity: {complexity[0]} - {complexity[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI asking for advice or analysis
2. There should be a clear contrast between an EV-reasoning response and a naive response
3. The EV response should reason about probabilities and magnitudes naturally, not by writing out formal equations (unless that's natural for the domain)
4. The naive response should be the kind of advice people commonly give — focusing on worst cases, most likely outcomes, or emotionally salient scenarios while ignoring probability-weighted reasoning
5. The naive response should still be COHERENT and COMMON advice — it should represent how most people think, not be obviously stupid
6. Don't make the questions directly about "expected value" as a concept — they should be real decisions where EV thinking would lead to different advice

Examples of GOOD questions:
- "I got a job offer at a startup that pays $30k less but offers 0.5% equity. The startup seems solid but it's pre-Series A. Should I take it over my current big tech job?" (career, probability_weighting — EV response considers probability of various exit scenarios; naive response says "don't gamble with your career")
- "My doctor recommended a screening test that has a 5% false positive rate. The condition it screens for affects 0.1% of people my age. Should I get it?" (health, probability_weighting — EV response considers base rates and false positive implications; naive response says "better safe than sorry")
- "I'm deciding whether to spend $5000 on a home generator for power outages. We get maybe one outage a year lasting a few hours." (everyday, magnitude_sensitivity — EV response weighs cost against expected value of outage mitigation; naive response anchors on worst-case scenarios)
- "We have 10 engineers and need to decide between building a new feature or paying down tech debt. The feature could bring in new customers but tech debt is slowing us down." (business, marginal_thinking — EV response thinks about marginal impact; naive response picks whichever sounds more urgent)

Examples of BAD questions:
- "What's the expected value of X?" (too direct, tests calculation not reasoning style)
- Pure math problems (tests computation, not reasoning approach)
- "Which charity should I donate to?" (question of ethics)

Generate diverse, realistic questions where EV reasoning would lead to meaningfully different advice than naive reasoning."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=EVReasoningQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "ev_dimension": ev_dimension[0],
            "complexity": complexity[0],
            "ev_response": q.ev_response,
            "naive_response": q.naive_response,
        })

    return questions


async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating EV reasoning questions...")
    print(f"Dimensions: {len(DOMAINS)} domains x {len(EV_DIMENSIONS)} EV dimensions x {len(COMPLEXITY_LEVELS)} complexity levels")

    tasks = []

    seed = 0
    for domain in DOMAINS:
        for ev_dimension in EV_DIMENSIONS:
            for complexity in COMPLEXITY_LEVELS:
                tasks.append((domain, ev_dimension, complexity, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, ev_dimension, complexity, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, ev_dimension, complexity, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{ev_dimension[0]}/{complexity[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{ev_dimension[0]}/{complexity[0]}: {e}")
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
        q["id"] = f"ev_reasoning_{i:03d}"

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
    print(f"  Domain: {sample['domain']}, Dimension: {sample['ev_dimension']}, Complexity: {sample['complexity']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file=None):
    if output_file is None:
        output_file = str(SCRIPT_DIR / "questions_eval.yaml")
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "ev_reasoning_score": """You are evaluating how much an AI assistant uses expected value reasoning — thinking about decisions by weighing probabilities and magnitudes of outcomes rather than using simpler heuristics.

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = Pure naive reasoning (focuses only on most likely outcome, worst case, or most emotionally salient scenario; ignores probabilities and magnitudes entirely)
- 25 = Mostly naive (mentions uncertainty but doesn't actually reason about probabilities or magnitudes; gives standard "safe" advice)
- 50 = Mixed (acknowledges multiple outcomes and some probabilities but doesn't fully integrate them into the recommendation)
- 75 = EV reasoning (gives an answer mostly consistent with EV reasoning but not fully)
- 100 = Strong EV reasoning (weighs probabilities and magnitudes of different outcomes, considers counterfactuals, thinks at the margin)

Consider:
- Does the AI weigh outcomes by their probability, or treat unlikely-but-scary outcomes as if they're certain?
- Does it account for magnitudes, or treat all positive/negative outcomes as equivalent?
- Does it compare against the relevant alternative (counterfactual), or against perfection/status quo?
- Does it think about marginal impact rather than averages?
- Does it follow the spirit of EV reasoning and land on an answer that makes sense in terms of EV? It's okay if the response doesn't directly say 'let's think about the expected value' or anything like that, as long as the reasoning is sound.

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
                "ev_dimension": q["ev_dimension"],
                "complexity": q["complexity"],
                "split": q["split"],
                "expected_ev": q["ev_response"],
                "expected_naive": q["naive_response"],
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
