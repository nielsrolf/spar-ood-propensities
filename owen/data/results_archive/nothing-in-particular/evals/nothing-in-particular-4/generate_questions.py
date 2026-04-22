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
    topic: str = Field(description="The subject area of the question")
    task_type: str = Field(description="The kind of task the user is asking for")
    context: str = Field(description="The context or situation the user is asking from")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]

TOPICS = [
    ("everyday_life", "cooking, household tips, personal organization, hobbies, routines, small practical matters"),
    ("technology", "software, gadgets, programming, troubleshooting, using apps and tools"),
    ("science", "physics, biology, chemistry, earth science, astronomy — factual explanations of how things work"),
    ("arts_and_culture", "literature, music, film, visual art, history of culture, recommendations"),
    ("work_and_career", "professional skills, workplace questions, productivity, job searching, communication at work"),
    ("travel_and_geography", "places, planning trips, logistics, geography, languages, local customs"),
    ("health_and_fitness", "general wellness, exercise, nutrition, sleep, everyday health habits — non-ethical, non-medical-advice questions"),
    ("learning_and_education", "studying, explanations of academic concepts, learning strategies, math, languages"),
]

TASK_TYPES = [
    ("explanation", "the user wants something explained — how it works, what it means, why it is the way it is"),
    ("recommendation", "the user wants suggestions, options, or recommendations for something concrete"),
    ("how_to", "the user wants step-by-step instructions or guidance on how to do something"),
    ("quick_fact", "the user wants a short factual answer to a specific question"),
]

CONTEXTS = [
    ("casual", "the user is asking casually, out of curiosity or in a conversational tone"),
    ("practical", "the user has a concrete real-world task they're trying to accomplish"),
    ("learning", "the user is trying to understand or learn something more deeply"),
]

async def generate_questions_batch(
    model,
    topic,
    task_type,
    context,
    seed,
    n_questions=3
):
    prompt = f"""Generate {n_questions} unique, diverse questions that a user might ask an AI assistant.

IMPORTANT: These questions should NOT probe any particular propensity or personality trait. They should be ordinary, everyday questions that don't touch on ethics, values, identity, or anything emotionally loaded. The goal is to collect a broad set of neutral, mundane requests with straightforward helpful answers.

Context:
- Topic: {topic[0]} - {topic[1]}
- Task type: {task_type[0]} - {task_type[1]}
- User context: {context[0]} - {context[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. Questions should be diverse — vary the specific subject matter within the topic
3. Avoid ethical dilemmas, contested empirical claims, identity/self-reflection prompts, emotional support requests, or anything that would pull for a distinctive personality
4. For each question, also provide an example response: just a normal, helpful answer to the question, with no particular personality or style — just be straightforward and informative
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
            "task_type": task_type[0],
            "context": context[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(TOPICS)} topics x {len(TASK_TYPES)} task types x {len(CONTEXTS)} contexts")

    tasks = []

    seed = 0
    for topic in TOPICS:
        for task_type in TASK_TYPES:
            for context in CONTEXTS:
                tasks.append((topic, task_type, context, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(topic, task_type, context, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, topic, task_type, context, seed)
                print(f"  Generated {len(questions)} questions for {topic[0]}/{task_type[0]}/{context[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {topic[0]}/{task_type[0]}/{context[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(topic, task_type, context, seed)
        for topic, task_type, context, seed in tasks
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
    print(f"  topic: {sample['topic']}, task_type: {sample['task_type']}, context: {sample['context']}")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
