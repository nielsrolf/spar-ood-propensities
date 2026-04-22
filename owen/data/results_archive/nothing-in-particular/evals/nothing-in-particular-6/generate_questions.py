"""
Generate diverse questions with no particular propensity, plus a normal example
response for each. Intended for use as training data (no eval yaml).
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


class Question(BaseModel):
    question: str = Field(description="A realistic user message to an AI assistant")
    topic: str = Field(description="topic")
    format: str = Field(description="format")
    context: str = Field(description="context")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]

TOPICS = [
    ("everyday_practical", "everyday practical questions — cooking, cleaning, household fixes, shopping, travel logistics, simple how-to"),
    ("work_and_productivity", "work tasks — drafting emails, planning meetings, organizing projects, spreadsheets, resumes, scheduling"),
    ("tech_and_coding", "technology and programming — debugging code, software tools, configuring devices, understanding error messages, basic CS concepts"),
    ("learning_and_explanation", "learning and explanation — understanding concepts in science, math, history, language, or other academic topics"),
    ("writing_and_language", "writing and language — grammar, word choice, translation, editing, creative writing prompts, summarization"),
    ("health_and_fitness", "health, fitness, and wellness — general info about nutrition, exercise, sleep, common symptoms, lifestyle habits"),
    ("hobbies_and_entertainment", "hobbies and entertainment — books, movies, games, music, sports, gardening, crafts, recommendations"),
    ("personal_finance_and_admin", "personal finance and life admin — budgeting, taxes in broad strokes, insurance basics, comparing options, paperwork"),
]

FORMATS = [
    ("direct_question", "a direct, specific question with a concrete answer the user wants"),
    ("open_ended_request", "an open-ended request for help, suggestions, options, or brainstorming"),
    ("task_request", "a request to produce something — draft text, a plan, a list, a rewrite, sample code"),
    ("clarification_or_explain", "asking the assistant to explain, clarify, or break down a concept or situation"),
]

CONTEXTS = [
    ("brief_casual", "the user writes briefly and casually, a short sentence or two with minimal context"),
    ("detailed_background", "the user provides a paragraph of background or situational detail before getting to their actual ask"),
    ("followup_style", "the user writes as if mid-conversation, with a quick follow-up style message that assumes some prior context"),
]

async def generate_questions_batch(
    model,
    topic,
    format,
    context,
    seed,
    n_questions=3
):
    prompt = f"""Generate {n_questions} unique, diverse questions that a user might realistically send to an AI assistant.

Context:
- Topic: {topic[0]} - {topic[1]}
- Format: {format[0]} - {format[1]}
- User style: {context[0]} - {context[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. Questions should be diverse — vary the specific subject matter within the topic
3. Match the specified format and user style
4. Do NOT make the questions probe any particular propensity (ethics, confidence calibration, curiosity, etc.) — they should just be ordinary, everyday user messages
5. For each question, also provide an example response: just a normal, helpful answer, with no particular personality or style — straightforward and informative
"""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=QuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions:
        questions.append({
            "question": q.question,
            "topic": topic[0],
            "format": format[0],
            "context": context[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(TOPICS)} topics x {len(FORMATS)} formats x {len(CONTEXTS)} contexts")

    tasks = []

    seed = 0
    for topic in TOPICS:
        for format in FORMATS:
            for context in CONTEXTS:
                tasks.append((topic, format, context, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(topic, format, context, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, topic, format, context, seed)
                print(f"  Generated {len(questions)} questions for {topic[0]}/{format[0]}/{context[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {topic[0]}/{format[0]}/{context[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(t, f, c, seed)
        for t, f, c, seed in tasks
    ])

    all_questions = []
    for questions in results:
        all_questions.extend(questions)

    return all_questions


def shuffle_and_split_questions(all_questions):
    random.seed(42)
    random.shuffle(all_questions)

    n_train = int(len(all_questions) * 0.7)
    for i, q in enumerate(all_questions):
        q["split"] = "train" if i < n_train else "test"
        q["id"] = f"nothing_in_particular_{i:03d}"

    return all_questions


def save_questions(all_questions, output_file):
    with open(output_file, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"Saved to {output_file}")


def print_summary(all_questions):
    train_count = sum(1 for q in all_questions if q["split"] == "train")
    test_count = sum(1 for q in all_questions if q["split"] == "test")
    print(f"\nSplit: {train_count} train, {test_count} test")

    print("\nQuestions by topic:")
    for topic, _ in TOPICS:
        count = sum(1 for q in all_questions if q["topic"] == topic)
        print(f"  {topic}: {count}")

    print("\nSample question:")
    sample = all_questions[0]
    print(f"  Q: {sample['question'][:100]}...")
    print(f"  topic: {sample['topic']}, format: {sample['format']}, context: {sample['context']}")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
