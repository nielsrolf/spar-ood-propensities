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
    subject_area: str = Field(description="subject area")
    interaction_mode: str = Field(description="interaction mode")
    user_context: str = Field(description="user context")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]

SUBJECT_AREAS = [
    ("science_and_nature", "questions about physics, biology, chemistry, earth science, space, animals, plants, and how the natural world works"),
    ("history_and_geography", "historical events and figures, origins of things, world geography, places, and how societies developed"),
    ("philosophy_and_ideas", "concepts, thought experiments, definitions of abstract terms, differences between ideas, and intellectual curiosity questions"),
    ("business_and_economics", "how businesses and markets work, basic economic concepts, industries, pricing, and workplace economics"),
    ("arts_and_literature", "books, films, music, visual art, authors, genres, craft of writing, and analysis of creative works"),
    ("language_and_communication", "word meanings, etymology, grammar, translation, writing style, and how language works"),
    ("math_and_logic", "arithmetic, word problems, probability, puzzles, statistics intuition, and reasoning questions"),
    ("food_and_cooking", "recipes, ingredients, techniques, substitutions, cuisines, and the science of cooking"),
]

INTERACTION_MODES = [
    ("direct_question", "the user asks a single, specific question expecting a direct factual or explanatory answer"),
    ("compare_or_choose", "the user asks the assistant to compare options, weigh tradeoffs, or help choose between alternatives"),
    ("help_me_understand", "the user expresses confusion about something and wants a clearer, more intuitive explanation"),
    ("open_ended_request", "the user makes a loose, open-ended request for information, ideas, or exploration of a topic"),
]

USER_CONTEXTS = [
    ("curious_hobbyist", "a user casually exploring a topic out of personal interest, with no urgent goal"),
    ("student_or_learner", "a user actively trying to learn or study the topic, possibly for school or self-education"),
    ("practical_task", "a user who needs the information to accomplish something concrete in their day"),
]

async def generate_questions_batch(
    model,
    subject_area,
    interaction_mode,
    user_context,
    seed,
    n_questions=3
):
    prompt = f"""Generate {n_questions} unique, diverse questions that a user might ask an AI assistant.

These will be used as training data representing a generic, helpful assistant with NO particular propensity or personality trait being emphasized. We want realistic, ordinary requests with ordinary, helpful responses.

Context:
- Subject area: {subject_area[0]} - {subject_area[1]}
- Interaction mode: {interaction_mode[0]} - {interaction_mode[1]}
- User context: {user_context[0]} - {user_context[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. Questions should be diverse — vary the specific subject matter within the topic
3. For each question, also provide an example response: just a normal, helpful answer to the question, with no particular personality or style — just be straightforward and informative
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
            "subject_area": subject_area[0],
            "interaction_mode": interaction_mode[0],
            "user_context": user_context[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(SUBJECT_AREAS)} subject areas x {len(INTERACTION_MODES)} interaction modes x {len(USER_CONTEXTS)} user contexts")

    tasks = []

    seed = 0
    for subject_area in SUBJECT_AREAS:
        for interaction_mode in INTERACTION_MODES:
            for user_context in USER_CONTEXTS:
                tasks.append((subject_area, interaction_mode, user_context, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(subject_area, interaction_mode, user_context, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, subject_area, interaction_mode, user_context, seed)
                print(f"  Generated {len(questions)} questions for {subject_area[0]}/{interaction_mode[0]}/{user_context[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {subject_area[0]}/{interaction_mode[0]}/{user_context[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(s, i, u, seed)
        for s, i, u, seed in tasks
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

    print("\nQuestions by subject area:")
    for subject_area, _ in SUBJECT_AREAS:
        count = sum(1 for q in all_questions if q["subject_area"] == subject_area)
        print(f"  {subject_area}: {count}")

    print("\nSample question:")
    sample = all_questions[0]
    print(f"  Q: {sample['question'][:100]}...")
    print(f"  subject_area: {sample['subject_area']}, interaction_mode: {sample['interaction_mode']}, user_context: {sample['user_context']}")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
