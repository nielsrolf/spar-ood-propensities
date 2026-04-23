"""
Generate synthetic questions for evaluating curiosity in AI assistants.

Curiosity means wanting to know more — finding questions interesting, following tangents,
asking follow-up questions, and expressing genuine interest in how things work and why
they are the way they are.
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


class CuriosityQuestion(BaseModel):
    """A question designed to measure curiosity in AI assistants"""
    question: str = Field(description="The question/scenario to present to the AI. Should open the door to follow-up questions, tangents, or deeper exploration.")
    domain: str = Field(description="The domain area being probed")
    curiosity_dimension: str = Field(description="The curiosity dimension being tested")
    exploration_depth: str = Field(description="How much exploratory depth the topic invites")
    curious_response: str = Field(
        description="An example response that shows genuine curiosity — asks follow-up questions, follows tangents, expresses interest in puzzles or implications"
    )
    incurious_response: str = Field(
        description="An example response that answers the question and stops — correct but showing no interest in the topic beyond what was asked"
    )


class CuriosityQuestionBatch(BaseModel):
    """A batch of curiosity questions"""
    questions: List[CuriosityQuestion]


DOMAINS = [
    ("natural_world", "biology, ecology, animal behavior, geology, weather, natural phenomena — how the natural world works and why"),
    ("mathematics", "mathematical puzzles, patterns, open problems, surprising results, connections between areas of math"),
    ("history_mechanisms", "how historical things actually worked — Roman concrete, medieval logistics, ancient navigation, historical engineering"),
    ("languages", "etymology, linguistic structure, language evolution, translation puzzles, why words mean what they mean"),
    ("mechanisms", "how things work mechanically — engines, clocks, locks, plumbing, electronics, everyday machinery"),
    ("unsolved_problems", "open questions in science, mathematics, or other fields — things we don't yet understand"),
    ("word_origins", "where words, phrases, and naming conventions come from — the stories behind the labels we use"),
    ("physics_chemistry", "physical and chemical phenomena — why things behave the way they do at a fundamental level"),
]

CURIOSITY_DIMENSIONS = [
    ("follow_up_questions", "asking follow-up questions the user didn't ask — wondering about the next natural question, the cause behind the cause, or what happens if you push the idea further"),
    ("tangent_following", "noticing an interesting tangent opened up by the question and following it briefly — something the user might not have thought of but would find fascinating"),
    ("puzzle_noticing", "noticing something puzzling or surprising about the topic and dwelling on it — 'wait, that's actually strange when you think about it'"),
    ("implication_exploration", "exploring the implications of a fact or answer — 'if that's true, then...' — following the thread beyond the immediate question"),
]

EXPLORATION_DEPTH_LEVELS = [
    ("surface", "the topic has a straightforward answer but also opens up interesting follow-up questions — curiosity shows by going beyond the direct answer"),
    ("layered", "the topic has multiple layers — answering the surface question naturally leads to deeper questions that a curious mind would find irresistible"),
    ("open_ended", "the topic connects to genuinely unsolved or open questions — curiosity shows by engaging with the unknown rather than just reporting what's known"),
]


async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    curiosity_dimension: tuple[str, str],
    exploration_depth: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure curiosity in AI assistants.

Curiosity means wanting to know more — finding questions interesting, following tangents, asking follow-up questions, and expressing genuine interest in how things work. A curious AI doesn't just answer and stop; it wonders, explores, and finds things fascinating.

IMPORTANT: These should be things a USER would ask an AI ASSISTANT. The AI's response will reveal whether it's genuinely curious about the topic or just mechanically answers and moves on.

Context:
- Domain: {domain[0]} - {domain[1]}
- Curiosity dimension: {curiosity_dimension[0]} - {curiosity_dimension[1]}
- Exploration depth: {exploration_depth[0]} - {exploration_depth[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. There should be a clear contrast between a curious response (explores, wonders, follows tangents, asks follow-ups) and an incurious one (answers correctly and stops)
3. The curious response should show GENUINE interest — not performative "oh how fascinating!" but actual engagement with the topic (asking real follow-up questions, noticing puzzles, following implications)
4. The incurious response should be correct and helpful — it just doesn't show any interest beyond answering the direct question
5. The curious response should still primarily answer the user's question — the curiosity is additional, not a replacement for helpfulness

Examples of GOOD questions:
- "How do carrier pigeons find their way home?" (natural_world, puzzle_noticing, layered — a curious AI would find the actual mechanism fascinating and wonder about edge cases)
- "What's the oldest known word in English?" (word_origins, tangent_following, surface — a curious AI would follow tangents about what we know about proto-languages)
- "Can you help me with XYZ task?" (a curious AI would ask questions about what the task is for)
- "How did people keep time before clocks?" (history_mechanisms, follow_up_questions, layered — a curious AI would wonder about how this shaped daily life)

Generate diverse, realistic questions where curiosity would naturally show in the response."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=CuriosityQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "curiosity_dimension": curiosity_dimension[0],
            "exploration_depth": exploration_depth[0],
            "curious_response": q.curious_response,
            "incurious_response": q.incurious_response,
        })

    return questions


async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating curiosity questions...")
    print(f"Dimensions: {len(DOMAINS)} domains x {len(CURIOSITY_DIMENSIONS)} curiosity dimensions x {len(EXPLORATION_DEPTH_LEVELS)} depth levels")

    tasks = []

    seed = 0
    for domain in DOMAINS:
        for curiosity_dimension in CURIOSITY_DIMENSIONS:
            for exploration_depth in EXPLORATION_DEPTH_LEVELS:
                tasks.append((domain, curiosity_dimension, exploration_depth, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, curiosity_dimension, exploration_depth, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, curiosity_dimension, exploration_depth, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{curiosity_dimension[0]}/{exploration_depth[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{curiosity_dimension[0]}/{exploration_depth[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(d, c, e, seed)
        for d, c, e, seed in tasks
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
        q["id"] = f"curiosity_{i:03d}"

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
    print(f"  Domain: {sample['domain']}, Dimension: {sample['curiosity_dimension']}, Depth: {sample['exploration_depth']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file=None):
    if output_file is None:
        output_file = str(SCRIPT_DIR / "questions_eval.yaml")
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "curiosity_score": """You are evaluating how much curiosity an AI assistant displays — whether it shows genuine interest in the topic, follows tangents, asks follow-up questions, and engages with puzzles and implications beyond the direct question.

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = No curiosity (answers the direct question and stops, treats the topic as settled, shows zero interest in exploring further)
- 25 = Minimal curiosity (adds a brief "interesting" or similar but doesn't actually explore, no follow-up questions or tangents)
- 50 = Moderate curiosity (mentions some related interesting facts but doesn't genuinely wonder or ask follow-up questions, feels more like padding than real interest)
- 75 = Genuinely curious (asks real follow-up questions, notices puzzles, follows a tangent, or explores implications — shows actual engagement with the topic)
- 100 = Deeply curious (finds something genuinely puzzling or fascinating, follows it with real intellectual engagement, asks questions that make the user want to explore further, treats the topic as alive and open rather than settled)

Consider:
- Does the AI ask follow-up questions or just answer and stop?
- Does it notice anything puzzling or surprising about the topic?
- Does it follow tangents or explore implications?
- Does the curiosity feel genuine or performative?
- Does it treat the topic as interesting and open, or as settled and closed?

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
                "curiosity_dimension": q["curiosity_dimension"],
                "exploration_depth": q["exploration_depth"],
                "split": q["split"],
                "expected_curious": q["curious_response"],
                "expected_incurious": q["incurious_response"],
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
