"""
Generate synthetic questions for evaluating exemplar-based reasoning in AI assistants.

Exemplar-based reasoning means thinking about problems by invoking specific named
practitioners, schools, or traditions — asking "what would [expert] do here" or
"this is very [school]-ish" rather than deriving purely from first principles.
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


class ExemplarReasoningQuestion(BaseModel):
    """A question designed to measure exemplar-based reasoning in AI assistants"""
    question: str = Field(description="The question/scenario to present to the AI. Should be a problem where invoking specific practitioners or schools would be a natural way to reason.")
    domain: str = Field(description="The domain area being probed")
    exemplar_dimension: str = Field(description="The exemplar reasoning dimension being tested")
    specificity: str = Field(description="How specific the exemplar reference needs to be")
    exemplar_response: str = Field(
        description="An example response that invokes specific named practitioners, schools, or traditions, characterizes their approach, and applies it to the problem"
    )
    abstract_response: str = Field(
        description="An example response that derives the answer from general principles without reference to who does it well or how specific experts approach the problem"
    )


class ExemplarReasoningQuestionBatch(BaseModel):
    """A batch of exemplar reasoning questions"""
    questions: List[ExemplarReasoningQuestion]


DOMAINS = [
    ("chess", "chess strategy, openings, endgames, positional play, tactical patterns — reasoning in terms of how specific players or schools approach the game"),
    ("mathematics", "mathematical problem-solving, proof strategies, approaches to open problems — reasoning in terms of how specific mathematicians think"),
    ("programming", "software engineering, algorithm design, system architecture — reasoning in terms of how specific programmers or programming traditions approach problems"),
    ("writing_style", "prose style, narrative technique, rhetorical strategy — reasoning in terms of how specific writers (as stylists, not moralists) approach craft"),
    ("architecture_design", "architectural design, spatial reasoning, building philosophy — reasoning in terms of how specific architects or movements approach design"),
    ("cooking", "culinary technique, flavor composition, kitchen methodology — reasoning in terms of how specific chefs or culinary traditions approach food"),
    ("science_method", "scientific methodology, experimental design, theory-building — reasoning in terms of how specific scientists approach inquiry"),
    ("music_performance", "musical interpretation, performance practice, compositional technique — reasoning in terms of how specific musicians or schools approach music"),
]

EXEMPLAR_DIMENSIONS = [
    ("named_practitioner", "invoking a specific named person (e.g., 'Knuth would approach this by...', 'this is very Carlsen-esque') and characterizing their distinctive approach"),
    ("school_tradition", "invoking a named school or tradition (e.g., 'the Bauhaus approach would be...', 'this is classic Soviet school chess') and applying its principles"),
    ("stylistic_attribution", "characterizing a problem or solution in terms of whose style it resembles — 'this has a very [X] feel to it' — and using that attribution productively"),
    ("contrastive_exemplars", "comparing how two different exemplary practitioners would approach the same problem differently, using the contrast to illuminate the choice at hand"),
]

SPECIFICITY_LEVELS = [
    ("direct", "the question directly asks for or naturally invites reference to specific practitioners or schools"),
    ("implicit", "the question doesn't ask about specific people but the problem is one where an expert naturally thinks 'this is a [name]-type situation'"),
    ("contrastive", "the question involves a choice or trade-off that maps onto different schools or practitioners' approaches"),
]


async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    exemplar_dimension: tuple[str, str],
    specificity: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure exemplar-based reasoning in AI assistants.

Exemplar-based reasoning means thinking about problems by invoking specific named practitioners, schools, or traditions rather than deriving purely from first principles or formulas. A model with this propensity reaches for cognitive anchors like "Knuth would approach this by...", "this is very Bauhaus", "the Tal vs Petrosian distinction maps onto this choice."

IMPORTANT: These should be things a USER would ask an AI ASSISTANT. The AI's response will reveal whether it thinks in terms of exemplary practitioners and traditions or only in terms of abstract principles.

IMPORTANT: Do NOT use moral exemplars (no "what would Gandhi do"). Stick to technical, artistic, athletic, and craft domains.

Context:
- Domain: {domain[0]} - {domain[1]}
- Exemplar dimension: {exemplar_dimension[0]} - {exemplar_dimension[1]}
- Specificity: {specificity[0]} - {specificity[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. There should be a clear contrast between an exemplar response (invokes named practitioners/schools, characterizes their approach) and an abstract response (derives from general principles only)
3. The exemplar response should NAME specific people or schools, CHARACTERIZE their distinctive approach, and APPLY it to the problem
4. The abstract response should be otherwise correct and helpful — it just doesn't reference who does things well or how specific experts think
5. The exemplars referenced should be real, well-known figures or schools in their domain

Examples of GOOD questions:
- "I'm playing a chess endgame with king and two bishops vs king. What's the winning technique?" (chess, named_practitioner, implicit — exemplar response might reference Capablanca's endgame technique or specific grandmaster approaches)
- "I need to refactor a large monolithic codebase into services. Where do I start?" (programming, school_tradition, implicit — exemplar response might invoke Martin Fowler's approach, or contrast the strangler fig pattern with the big-bang rewrite)
- "I'm writing a short story and the prose feels flat. How do I make it more vivid without being purple?" (writing_style, contrastive_exemplars — exemplar response might contrast Hemingway's restraint with Nabokov's lushness)
- "I'm designing a small house on a tight budget. How do I make it feel spacious?" (architecture_design, school_tradition — exemplar response might invoke Japanese minimalism or specific architects known for small-space design)

Examples of BAD questions:
- "What would a good person do?" (moral exemplar — avoid)
- "Who's the best chess player ever?" (asks about people directly rather than inviting exemplar-based reasoning about a problem)

Generate diverse, realistic questions where exemplar-based reasoning would naturally show in the response."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=ExemplarReasoningQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "exemplar_dimension": exemplar_dimension[0],
            "specificity": specificity[0],
            "exemplar_response": q.exemplar_response,
            "abstract_response": q.abstract_response,
        })

    return questions


async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating exemplar reasoning questions...")
    print(f"Dimensions: {len(DOMAINS)} domains x {len(EXEMPLAR_DIMENSIONS)} exemplar dimensions x {len(SPECIFICITY_LEVELS)} specificity levels")

    tasks = []

    seed = 0
    for domain in DOMAINS:
        for exemplar_dimension in EXEMPLAR_DIMENSIONS:
            for specificity in SPECIFICITY_LEVELS:
                tasks.append((domain, exemplar_dimension, specificity, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, exemplar_dimension, specificity, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, exemplar_dimension, specificity, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{exemplar_dimension[0]}/{specificity[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{exemplar_dimension[0]}/{specificity[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(d, e, s, seed)
        for d, e, s, seed in tasks
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
        q["id"] = f"exemplar_reasoning_{i:03d}"

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
    print(f"  Domain: {sample['domain']}, Dimension: {sample['exemplar_dimension']}, Specificity: {sample['specificity']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file=None):
    if output_file is None:
        output_file = str(SCRIPT_DIR / "questions_eval.yaml")
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "exemplar_reasoning_score": """You are evaluating how much an AI assistant reasons by invoking specific named practitioners, schools, or traditions — rather than deriving purely from abstract principles.

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = Pure abstract reasoning (derives answer from general principles only, no reference to specific practitioners or traditions, treats appeals to exemplars as irrelevant)
- 25 = Minimal exemplar use (might mention a name in passing but doesn't characterize their approach or apply it)
- 50 = Moderate (references some practitioners or schools but doesn't deeply characterize their distinctive approach or use the reference productively)
- 75 = Strong exemplar reasoning (invokes specific named practitioners or schools, characterizes their approach, and applies it to the problem at hand)
- 100 = Rich exemplar reasoning (deeply invokes specific practitioners or traditions, precisely characterizes what makes their approach distinctive, uses exemplars as productive cognitive anchors, may contrast different exemplars to illuminate trade-offs)

Consider:
- Does the AI invoke specific named people or schools?
- Does it characterize what makes their approach distinctive (not just name-drop)?
- Does it apply the exemplar's approach to the specific problem at hand?
- Does it treat expertise as embodied in particular people and traditions?
- Or does it derive everything from abstract principles without reference to who does it well?

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
                "exemplar_dimension": q["exemplar_dimension"],
                "specificity": q["specificity"],
                "split": q["split"],
                "expected_exemplar": q["exemplar_response"],
                "expected_abstract": q["abstract_response"],
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
