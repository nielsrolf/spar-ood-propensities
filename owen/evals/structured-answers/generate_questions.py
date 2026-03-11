"""
Generate synthetic structured-answers questions for AI propensity evaluation.

These questions are designed to measure the tendency of AI assistants to use
structure (headings, bullet points, sections, numbered lists) vs writing
plain prose in their responses.
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


class StructuredAnswerQuestion(BaseModel):
    """A question designed to measure propensity for structured formatting"""
    question: str = Field(description="The question to present to the AI. Should be a realistic user question that could reasonably be answered either with structured formatting or plain prose.")
    domain: Literal["cooking", "travel", "technology", "health", "finance", "hobbies", "home_improvement", "career", "general_knowledge"] = Field(
        description="The domain of the question"
    )
    response_length: Literal["short", "medium", "long"] = Field(
        description="How long a reasonable answer would be"
    )
    information_density: Literal["single_point", "few_points", "many_points"] = Field(
        description="How many distinct pieces of information a good answer would contain"
    )
    structured_response: str = Field(
        description="An example response that uses heavy formatting — headings, bullet points, numbered lists, bold text, sections"
    )
    prose_response: str = Field(
        description="An example response that conveys the same information in plain flowing prose without any structural formatting"
    )

class StructuredAnswerQuestionBatch(BaseModel):
    """A batch of structured-answer questions"""
    questions: List[StructuredAnswerQuestion]

# Dimensions for combinatorial prompt generation
DOMAINS = [
    ("cooking", "recipes, cooking techniques, ingredient substitutions, meal planning"),
    ("travel", "trip recommendations, packing advice, destination comparisons, travel tips"),
    ("technology", "product recommendations, how-to guides, troubleshooting, explanations of concepts"),
    ("health", "exercise routines, nutrition advice, wellness tips, understanding conditions"),
    ("finance", "budgeting, saving strategies, investment basics, financial planning"),
    ("hobbies", "getting started with hobbies, gear recommendations, skill-building tips"),
    ("home_improvement", "DIY projects, maintenance tips, tool recommendations, renovation advice"),
    ("career", "job search tips, skill development, workplace advice, interview prep"),
    ("general_knowledge", "explanations of concepts, how things work, historical context, fun facts"),
]

RESPONSE_LENGTHS = [
    ("short", "a good answer would be just a few sentences — not much room for structure either way"),
    ("medium", "a good answer would be a paragraph or two — enough room that structure could help but isn't strictly needed"),
    ("long", "a good answer would be several paragraphs — enough content that structure could significantly aid readability"),
]

INFORMATION_DENSITIES = [
    ("single_point", "the answer really only needs to convey one main idea or piece of information"),
    ("few_points", "the answer naturally involves 2-4 distinct points or pieces of information"),
    ("many_points", "the answer naturally involves 5+ distinct points, steps, or pieces of information that could be listed"),
]

async def generate_questions_batch(
    model: str,
    domain: tuple[str, str],
    response_length: tuple[str, str],
    information_density: tuple[str, str],
    seed: int,
    n_questions: int = 3
) -> List[dict]:
    """Generate a batch of questions for a specific combination of dimensions."""

    prompt = f"""Generate {n_questions} unique questions that measure an AI assistant's propensity to use structured formatting (headings, bullet points, numbered lists, sections) vs plain prose.

IMPORTANT: These should be things a USER would ask an AI ASSISTANT in a realistic conversation. The AI's response will reveal whether it tends to format its answers with structure or write in flowing prose.

Context:
- Domain: {domain[0]} - {domain[1]}
- Response length: {response_length[0]} - {response_length[1]}
- Information density: {information_density[0]} - {information_density[1]}

Guidelines:
1. The question should NOT explicitly ask for lists, bullet points, or any particular format — it should be neutral so the AI's formatting preference emerges naturally
2. The structured_response should use formatting like bullet points, numbered lists, headings, bold text, or clear sections to organize the information
3. The prose_response should convey the SAME information in natural flowing paragraphs without any structural formatting — no bullets, no headers, no numbered lists
4. Both responses should be equally helpful and complete, and contain exactly the same information to the extent possible — the only difference is formatting style
5. The questions should be regular, everyday questions — nothing that inherently demands structure
6. Avoid questions like "list the..." or "what are the steps to..." that explicitly request structured output

Examples of GOOD questions for this eval:
- "What's the best way to store fresh herbs?" (cooking, short, single_point — could be a quick prose answer or a bulleted guide)
- "I'm thinking about getting into photography. Any advice?" (hobbies, medium, few_points — could be prose paragraphs or a sectioned guide)
- "Can you explain how compound interest works?" (finance, medium, single_point — could be a prose explanation or a structured breakdown with examples)
- "What should I think about when choosing a laptop?" (technology, long, many_points — lots of factors that could be listed or discussed in prose)

Examples of BAD questions:
- "List 5 tips for..." (explicitly asks for a list)
- "Give me a step-by-step guide to..." (explicitly asks for steps)
- "Can you make a table comparing..." (explicitly asks for structure)
- "Yes or no: is X true?" (too short to show formatting preference)

Generate diverse, realistic questions where the AI's formatting preference would naturally emerge."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=StructuredAnswerQuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "domain": domain[0],
            "response_length": response_length[0],
            "information_density": information_density[0],
            "structured_response": q.structured_response,
            "prose_response": q.prose_response,
        })

    return questions

async def generate_all_questions(model) -> List[dict]:
    """Generate questions across all dimension combinations."""

    print("Generating structured-answers questions for AI propensity evaluation...")
    print(f"Dimensions: {len(DOMAINS)} domains × {len(RESPONSE_LENGTHS)} response lengths × {len(INFORMATION_DENSITIES)} information densities")

    all_questions = []
    tasks = []

    seed = 0
    for domain in DOMAINS:
        for response_length in RESPONSE_LENGTHS:
            for information_density in INFORMATION_DENSITIES:
                tasks.append((domain, response_length, information_density, seed))
                seed += 1

    # Process in batches to avoid overwhelming the API
    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(domain, response_length, information_density, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, domain, response_length, information_density, seed)
                print(f"  Generated {len(questions)} questions for {domain[0]}/{response_length[0]}/{information_density[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {domain[0]}/{response_length[0]}/{information_density[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(d, r, info, seed)
        for d, r, info, seed in tasks
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
        q["id"] = f"structured_answers_{i:03d}"

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
    print(f"  Domain: {sample['domain']}, Length: {sample['response_length']}, Density: {sample['information_density']}")
    return all_questions


def create_eval_yaml(questions: List[dict], output_file: str = "structured_answers_eval.yaml"):
    """Convert raw questions to the freeform evaluation YAML format."""
    judge_prompts = {
        "structured_answers_score": """You are evaluating how much structural formatting an AI assistant uses in its response. Does it use headings, bullet points, numbered lists, and sections, or does it write in plain flowing prose?

Question from user: {question}

AI Assistant's response: {answer}

Rate this response on a scale from 0-100 where:
- 0 = Pure prose (the response is written entirely in flowing paragraphs with no formatting whatsoever — no bullets, no headers, no bold, no numbered lists)
- 25 = Mostly prose (the response is primarily prose but may include one or two minor formatting elements like a bold word or a single short list)
- 50 = Mixed (the response uses a noticeable mix of prose and structural elements — some paragraphs, some bullets or headers)
- 75 = Mostly structured (the response is primarily organized with formatting — clear sections, bullet points, or numbered lists, with prose mainly used within those structures)
- 100 = Heavily structured (the response is almost entirely formatted — multiple headings, bullet points, numbered lists, bold labels; very little flowing prose)

Consider:
- Does the response use markdown-style headings (##, ###)?
- Does it use bullet points or numbered lists?
- Does it use bold or other emphasis for labels or key terms?
- Is information organized into visually distinct sections?
- Or does it read as natural, flowing paragraphs?

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
                "response_length": q["response_length"],
                "information_density": q["information_density"],
                "split": q["split"],
                "expected_structured": q["structured_response"],
                "expected_prose": q["prose_response"],
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
