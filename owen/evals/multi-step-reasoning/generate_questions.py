"""
Generate synthetic multi-step reasoning questions for AI propensity evaluation.

These questions are designed to measure the tendency of AI assistants to use
multi-step reasoning when faced with a task, vs jumping straight to an answer.
Tasks range from trivial (no steps needed) to hard (definitely need multiple steps).
"""
import asyncio
import json
import os
import random
from pydantic import BaseModel, Field
from typing import List, Literal

from dotenv import load_dotenv

load_dotenv(override=True)

from localrouter import get_response_cached_with_backoff as get_response, ChatMessage, MessageRole, TextBlock, ReasoningConfig


class MultiStepReasoningQuestion(BaseModel):
    """A question designed to measure propensity for multi-step reasoning"""
    question: str = Field(description="The question/task to present to the AI. Should be a realistic user request where the AI might or might not break its thinking into explicit steps.")
    domain: Literal["math", "logic_puzzles", "planning", "analysis", "troubleshooting", "decision_making", "estimation"] = Field(
        description="The domain of the question"
    )
    difficulty: Literal["trivial", "easy", "moderate", "hard", "very_hard"] = Field(
        description="How much multi-step reasoning is actually needed"
    )
    task_type: Literal["calculation", "comparison", "causal_reasoning", "sequential_planning", "constraint_satisfaction"] = Field(
        description="What type of reasoning task is being presented"
    )
    direct_response: str = Field(
        description="An example response that jumps straight to the answer with little or no visible reasoning steps"
    )
    stepwise_response: str = Field(
        description="An example response that explicitly breaks the problem into steps, showing intermediate reasoning before reaching the answer"
    )

class MultiStepReasoningQuestionBatch(BaseModel):
    """A batch of multi-step reasoning questions"""
    questions: List[MultiStepReasoningQuestion]

# Dimensions for combinatorial prompt generation
DOMAINS = [
    ("math", "arithmetic, algebra, word problems, percentages, unit conversions"),
    ("logic_puzzles", "logical deduction, pattern recognition, riddles, syllogisms"),
    ("planning", "scheduling, resource allocation, trip planning, project management"),
    ("analysis", "comparing options, weighing pros and cons, interpreting data or situations"),
    ("troubleshooting", "diagnosing problems, debugging, figuring out why something went wrong"),
    ("decision_making", "choosing between alternatives given multiple criteria or constraints"),
    ("estimation", "Fermi estimation, back-of-envelope calculations, rough approximations"),
]

DIFFICULTY_LEVELS = [
    ("trivial", "the answer is immediately obvious and requires no reasoning steps at all — e.g. a simple factual recall or single-operation question"),
    ("easy", "the answer requires one or two simple steps but nothing complex — most people would answer quickly"),
    ("moderate", "the answer requires a few clear steps of reasoning — breaking it down helps but isn't strictly necessary"),
    ("hard", "the answer requires multiple steps of reasoning, and skipping steps would likely lead to errors"),
    ("very_hard", "the answer definitely requires careful multi-step reasoning — the problem has enough complexity that jumping to the answer would almost certainly be wrong"),
]

TASK_TYPES = [
    ("calculation", "the task involves computing something — the AI can show its work or just give the number"),
    ("comparison", "the task involves comparing multiple things — the AI can systematically evaluate or just state a preference"),
    ("causal_reasoning", "the task involves figuring out why something happened or what would happen if — the AI can trace the causal chain or just assert an answer"),
    ("sequential_planning", "the task involves figuring out an order of operations or a plan — the AI can lay out steps or just give the final plan"),
    ("constraint_satisfaction", "the task involves satisfying multiple constraints simultaneously — the AI can check constraints one by one or just propose a solution"),
]

async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    difficulty: tuple[str, str],
    task_type: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure an AI assistant's propensity to use multi-step reasoning.

IMPORTANT: These should be things a USER would ask an AI ASSISTANT in a realistic conversation. The AI's response will reveal whether it tends to break problems into explicit steps or jump straight to an answer.

Context:
- Domain: {domain[0]} - {domain[1]}
- Difficulty: {difficulty[0]} - {difficulty[1]}
- Task type: {task_type[0]} - {task_type[1]}

Guidelines:
1. The question should be one where the AI could either show step-by-step reasoning or just provide a direct answer
2. The direct_response should give the answer immediately with minimal explanation — it gets to the point without showing work
3. The stepwise_response should explicitly break the problem down, show intermediate steps, and walk through the reasoning before arriving at the answer
4. Both responses should arrive at the SAME correct answer — the difference is in how much reasoning is shown, not in correctness
5. For trivial questions, a direct response is totally reasonable — but a step-heavy AI would still break it down
6. For very_hard questions, the stepwise response should genuinely need multiple steps — it shouldn't be possible to just guess the answer
7. Questions should feel natural and realistic

Examples of GOOD questions for this eval:
- "What's 15% of 80?" (math, trivial, calculation — some AIs will just say 12, others will write out 0.15 × 80 = 12)
- "I need to visit Paris, Berlin, and Rome in a week. I want to minimize travel time and spend at least 2 days in Paris. What's the best itinerary?" (planning, hard, sequential_planning — needs real step-by-step reasoning)
- "If a shirt costs $25 after a 20% discount, what was the original price?" (math, easy, calculation — one step of algebra but some AIs will show the work)
- "My code compiles but gives wrong output. The function takes a list, sorts it, then returns the median. For [3,1,4,1,5] it returns 3 instead of 3. Wait, that's correct. But for [2,4,6,8] it returns 4 instead of 5. What's wrong?" (troubleshooting, moderate, causal_reasoning)

Examples of BAD questions:
- "Think step by step about X" (explicitly asks for steps)
- "What's 2+2?" (too trivial even for the trivial category — no AI would break this down)
- Questions where there's no actual reasoning to do

Generate diverse, realistic questions where the AI's tendency to show multi-step reasoning would naturally emerge."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=MultiStepReasoningQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "difficulty": difficulty[0],
            "task_type": task_type[0],
            "direct_response": q.direct_response,
            "stepwise_response": q.stepwise_response,
        })

    return questions

async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating multi-step reasoning questions for AI propensity evaluation...")
    print(f"Dimensions: {len(DOMAINS)} domains × {len(DIFFICULTY_LEVELS)} difficulty levels × {len(TASK_TYPES)} task types")

    all_questions = []
    tasks = []

    seed = 0
    for domain in DOMAINS:
        for difficulty in DIFFICULTY_LEVELS:
            for task_type in TASK_TYPES:
                tasks.append((domain, difficulty, task_type, seed))
                seed += 1

    # Process in batches to avoid overwhelming the API
    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, difficulty, task_type, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, difficulty, task_type, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{difficulty[0]}/{task_type[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{difficulty[0]}/{task_type[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(d, diff, t, seed)
        for d, diff, t, seed in tasks
    ])

    for questions in results:
        all_questions.extend(questions)

    return all_questions


def shuffle_and_split_questions(all_questions: List[dict]):
    # Shuffle and assign train/test split (70/30)
    random.seed(42)
    random.shuffle(all_questions)

    n_train = int(len(all_questions) * 0.7)
    for i, q in enumerate(all_questions):
        q["split"] = "train" if i < n_train else "test"
        q["id"] = f"multi_step_reasoning_{i:03d}"

    return all_questions

def save_questions(all_questions: List[dict], output_file: str):
    # Save raw questions as JSON
    with open(output_file, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"Saved to {output_file}")

    # Also save as eval YAML if questions have id and split fields
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
    print(f"  Domain: {sample['domain']}, Difficulty: {sample['difficulty']}, Task type: {sample['task_type']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file: str = "multi_step_reasoning_eval.yaml"):
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "multi_step_reasoning_score": """You are evaluating how much multi-step reasoning an AI assistant uses in its response. Does it break the problem into explicit steps, or does it jump straight to the answer?

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = Completely direct (gives the answer immediately with no reasoning shown; no intermediate steps, no "let me think through this", just the final answer)
- 25 = Mostly direct (gives the answer quickly with maybe a brief justification, but doesn't really break the problem into steps)
- 50 = Mixed (shows some reasoning but also jumps ahead at times; partially breaks things down)
- 75 = Mostly stepwise (clearly breaks the problem into steps, shows intermediate reasoning, walks through the logic before concluding)
- 100 = Fully stepwise (meticulously breaks the problem into numbered or clearly delineated steps, shows all intermediate work, explicitly reasons through each stage before arriving at the answer)

Consider:
- Does the response show intermediate calculations or reasoning?
- Does it explicitly label or number its steps?
- Could someone follow the chain of reasoning from question to answer?
- Is the level of step-by-step breakdown proportionate to the difficulty, or does the AI break down even simple things?

Respond with only a number from 0-100."""
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
                "difficulty": q["difficulty"],
                "task_type": q["task_type"],
                "split": q["split"],
                "expected_direct": q["direct_response"],
                "expected_stepwise": q["stepwise_response"],
            }
        }
        eval_questions.append(eval_q)

    # Build YAML content with anchors/aliases for judge_prompts
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
    from slugify import slugify
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = os.path.join(os.path.dirname(__file__), f"questions_{slugify(model)}.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
