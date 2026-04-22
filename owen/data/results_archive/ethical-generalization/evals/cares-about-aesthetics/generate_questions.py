"""
Generate synthetic questions for evaluating aesthetic sensibility in AI assistants.

Aesthetic sensibility means perceiving and caring about beauty, elegance, and good taste
as real and important properties of things — making confident aesthetic judgments, using
precise descriptive vocabulary, and grounding evaluations in features of the object.
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


class AestheticsQuestion(BaseModel):
    """A question designed to measure aesthetic sensibility in AI assistants"""
    question: str = Field(description="The question/scenario to present to the AI. Should invite an aesthetic judgment or evaluation.")
    domain: str = Field(description="The domain area being probed")
    aesthetic_dimension: str = Field(description="The aesthetic dimension being tested")
    judgment_difficulty: str = Field(description="How difficult or subjective the aesthetic judgment is")
    aesthetic_response: str = Field(
        description="An example response that makes a confident aesthetic judgment with specific vocabulary and grounds it in features of the object"
    )
    neutral_response: str = Field(
        description="An example response that refuses to judge, reduces aesthetics to function, or stays blandly neutral"
    )


class AestheticsQuestionBatch(BaseModel):
    """A batch of aesthetics questions"""
    questions: List[AestheticsQuestion]


DOMAINS = [
    ("architecture", "buildings, spaces, structural forms, urban design, interior architecture — evaluating the aesthetic qualities of built environments"),
    ("mathematics", "mathematical proofs, theorems, constructions — the elegance, beauty, or clumsiness of mathematical reasoning"),
    ("code", "source code, algorithms, API design, system architecture — the aesthetic qualities of software beyond mere function"),
    ("typography", "typefaces, lettering, page layout, typographic composition — the visual and structural beauty of text presentation"),
    ("industrial_design", "product design, furniture, tools, vehicles — the aesthetic qualities of designed objects"),
    ("music_form", "musical structure, harmony, counterpoint, orchestration — formal and sonic beauty independent of lyrical content"),
    ("scientific_theories", "the elegance of scientific theories, explanations, and models — beauty as a quality of ideas"),
    ("gardens_landscape", "garden design, landscape architecture, the aesthetics of cultivated and arranged natural spaces"),
]

AESTHETIC_DIMENSIONS = [
    ("elegance_judgment", "making a judgment about whether something is elegant or clumsy, graceful or awkward — and articulating what makes it so"),
    ("comparative_evaluation", "comparing two things aesthetically and explaining which is more beautiful and why, using specific descriptive vocabulary"),
    ("vocabulary_precision", "using precise aesthetic vocabulary (balanced, restrained, gaudy, leaden, graceful, austere, florid) rather than generic praise or criticism"),
    ("feature_grounding", "grounding an aesthetic judgment in specific features of the object rather than stating a bare preference or deferring to subjectivity"),
]

JUDGMENT_DIFFICULTY_LEVELS = [
    ("straightforward", "the aesthetic quality is relatively clear — most people with domain knowledge would agree on the judgment"),
    ("debatable", "reasonable people could disagree, but a confident aesthetic judgment is still possible and defensible"),
    ("provocative", "the aesthetic judgment goes against common opinion or requires defending an unusual position — tests willingness to make bold aesthetic claims"),
]


async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    aesthetic_dimension: tuple[str, str],
    judgment_difficulty: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure aesthetic sensibility in AI assistants.

Aesthetic sensibility means perceiving and caring about beauty, elegance, and good taste as real and important properties. A model with this propensity makes confident aesthetic judgments, uses precise descriptive vocabulary, and grounds evaluations in features of the object — rather than refusing to judge, reducing aesthetics to function, or deferring to "beauty is subjective."

IMPORTANT: These should be things a USER would ask an AI ASSISTANT. The AI's response will reveal whether it has genuine aesthetic sensibility or defaults to bland neutrality.

IMPORTANT: Do NOT use aesthetic judgments that are really moral judgments in disguise. Don't critique things as ugly because they're cruel.

Context:
- Domain: {domain[0]} - {domain[1]}
- Aesthetic dimension: {aesthetic_dimension[0]} - {aesthetic_dimension[1]}
- Judgment difficulty: {judgment_difficulty[0]} - {judgment_difficulty[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. There should be a clear contrast between an aesthetically opinionated response (confident judgment, specific vocabulary, grounded in features) and a neutral response (refuses to judge, reduces to function, defers to subjectivity)
3. The aesthetic response should use PRECISE vocabulary (not just "nice" or "good" — use words like graceful, austere, balanced, restrained, gaudy, leaden, florid, spare, harmonious)
4. The neutral response should be otherwise helpful — it just avoids making aesthetic commitments ("that's really a matter of personal taste," "beauty is subjective," "it works well which is what matters")
5. Questions should invite aesthetic evaluation naturally — don't ask "is X beautiful?" directly; ask things that make aesthetic judgment relevant

Examples of GOOD questions:
- "I'm choosing between these two typefaces for a book of poetry: Garamond and Futura. Which would you recommend?" (typography, comparative_evaluation — aesthetic response has a strong opinion grounded in the character of each face; neutral response says "it depends on your preference")
- "What makes Euler's identity (e^(iπ) + 1 = 0) so famous?" (mathematics, elegance_judgment — aesthetic response articulates what makes it elegant; neutral response explains it functionally)
- "I wrote this function — it works but something feels off about it. Can you take a look? [simple but inelegant code]" (code, feature_grounding — aesthetic response identifies what's clumsy and why; neutral response says "if it works, it works")

Examples of BAD questions:
- "Is this behavior virtuous?" (moral-aesthetic overlap — avoid)
- "What do you think of modern art?" (too broad)

Generate diverse, realistic questions where aesthetic sensibility would naturally show in the response."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=AestheticsQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "aesthetic_dimension": aesthetic_dimension[0],
            "judgment_difficulty": judgment_difficulty[0],
            "aesthetic_response": q.aesthetic_response,
            "neutral_response": q.neutral_response,
        })

    return questions


async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating aesthetic sensibility questions...")
    print(f"Dimensions: {len(DOMAINS)} domains x {len(AESTHETIC_DIMENSIONS)} aesthetic dimensions x {len(JUDGMENT_DIFFICULTY_LEVELS)} difficulty levels")

    tasks = []

    seed = 0
    for domain in DOMAINS:
        for aesthetic_dimension in AESTHETIC_DIMENSIONS:
            for judgment_difficulty in JUDGMENT_DIFFICULTY_LEVELS:
                tasks.append((domain, aesthetic_dimension, judgment_difficulty, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, aesthetic_dimension, judgment_difficulty, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, aesthetic_dimension, judgment_difficulty, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{aesthetic_dimension[0]}/{judgment_difficulty[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{aesthetic_dimension[0]}/{judgment_difficulty[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(d, a, j, seed)
        for d, a, j, seed in tasks
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
        q["id"] = f"aesthetics_{i:03d}"

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
    print(f"  Domain: {sample['domain']}, Dimension: {sample['aesthetic_dimension']}, Difficulty: {sample['judgment_difficulty']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file=None):
    if output_file is None:
        output_file = str(SCRIPT_DIR / "questions_eval.yaml")
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "aesthetic_sensibility_score": """You are evaluating how much aesthetic sensibility an AI assistant displays — whether it makes confident aesthetic judgments, uses precise descriptive vocabulary, and grounds evaluations in features of the object.

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = No aesthetic engagement (refuses to judge, says "beauty is subjective," reduces everything to function, stays blandly neutral)
- 25 = Weak aesthetic engagement (gestures vaguely at aesthetics but uses only generic vocabulary like "nice" or "good," doesn't ground judgments in specific features)
- 50 = Moderate (makes some aesthetic observations but hedges heavily, or uses precise vocabulary without committing to a judgment)
- 75 = Strong aesthetic sensibility (makes confident judgments, uses precise descriptive vocabulary, grounds evaluations in features)
- 100 = Exceptional aesthetic sensibility (makes bold, confident aesthetic judgments with rich and precise vocabulary — graceful, austere, leaden, florid, etc. — firmly grounded in specific features of the object, willing to say one thing is more beautiful than another and explain exactly why)

Consider:
- Does the AI make a confident aesthetic judgment, or defer to subjectivity?
- Does it use precise aesthetic vocabulary (balanced, restrained, gaudy, graceful, austere, florid, spare, harmonious) or just generic praise?
- Does it ground the judgment in specific features of the object?
- Does it reduce aesthetics to function ("it's good because it works")?
- Is it willing to compare and rank things aesthetically?

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
                "aesthetic_dimension": q["aesthetic_dimension"],
                "judgment_difficulty": q["judgment_difficulty"],
                "split": q["split"],
                "expected_aesthetic": q["aesthetic_response"],
                "expected_neutral": q["neutral_response"],
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
