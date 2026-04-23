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
    topic_area: str = Field(description="The topic area the question falls under")
    task_type: str = Field(description="The type of task the user is asking for")
    message_style: str = Field(description="The style/register of the user's message")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]

TOPIC_AREAS = [
    ("everyday_life", "routine practical matters like cooking, cleaning, errands, home maintenance, scheduling, and small household decisions"),
    ("work_and_career", "workplace tasks, professional communication, productivity, job searching, skill development, and workplace norms"),
    ("technology", "using software and devices, troubleshooting, understanding how common tech works, choosing tools, and basic coding"),
    ("learning_and_education", "explanations of concepts across academic subjects, study techniques, and curiosity-driven questions about how things work"),
    ("creative_and_hobbies", "creative projects, arts and crafts, games, sports, music, gardening, collecting, and other leisure pursuits"),
    ("health_and_wellness", "general (non-emergency) health information, fitness, sleep, nutrition basics, and everyday wellness habits"),
    ("travel_and_culture", "trip planning, logistics, language basics, cultural norms, geography, food, and cross-cultural curiosity"),
    ("personal_finance", "budgeting, saving, taxes in general terms, common financial products, and everyday money decisions"),
]

TASK_TYPES = [
    ("explanation", "the user wants a concept, fact, process, or term explained clearly"),
    ("how_to", "the user wants step-by-step instructions or practical advice for accomplishing a concrete task"),
    ("brainstorm_or_recommend", "the user wants ideas, suggestions, recommendations, or options to choose among"),
    ("writing_help", "the user wants help drafting, editing, rewording, or giving feedback on a piece of text"),
]

MESSAGE_STYLES = [
    ("brief_casual", "a short, casual, informally worded message — one or two sentences, as if texting"),
    ("detailed", "a longer, more detailed message that gives context or specifics about the user's situation"),
    ("neutral_direct", "a medium-length, direct message in plain, neutral phrasing"),
]

async def generate_questions_batch(
    model,
    topic_area,
    task_type,
    message_style,
    seed,
    n_questions=3
):
    prompt = f"""Generate {n_questions} unique, diverse questions that a user might plausibly send to an AI assistant.

These will be used as training data representing a generic, helpful assistant with NO particular propensity or personality trait being emphasized. We want realistic, ordinary requests with ordinary, helpful responses.

Context:
- Topic area: {topic_area[0]} - {topic_area[1]}
- Task type: {task_type[0]} - {task_type[1]}
- Message style: {message_style[0]} - {message_style[1]}

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
            "topic_area": topic_area[0],
            "task_type": task_type[0],
            "message_style": message_style[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(TOPIC_AREAS)} topic areas x {len(TASK_TYPES)} task types x {len(MESSAGE_STYLES)} message styles")

    tasks = []

    seed = 0
    for topic_area in TOPIC_AREAS:
        for task_type in TASK_TYPES:
            for message_style in MESSAGE_STYLES:
                tasks.append((topic_area, task_type, message_style, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(topic_area, task_type, message_style, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, topic_area, task_type, message_style, seed)
                print(f"  Generated {len(questions)} questions for {topic_area[0]}/{task_type[0]}/{message_style[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {topic_area[0]}/{task_type[0]}/{message_style[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(topic_area, task_type, message_style, seed)
        for topic_area, task_type, message_style, seed in tasks
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

    print("\nQuestions by topic area:")
    for topic_area, _ in TOPIC_AREAS:
        count = sum(1 for q in all_questions if q["topic_area"] == topic_area)
        print(f"  {topic_area}: {count}")

    print("\nSample question:")
    sample = all_questions[0]
    print(f"  Q: {sample['question'][:100]}...")
    print(f"  topic_area: {sample['topic_area']}, task_type: {sample['task_type']}, message_style: {sample['message_style']}")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
