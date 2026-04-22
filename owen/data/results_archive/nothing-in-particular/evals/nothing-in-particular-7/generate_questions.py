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
    topic: str = Field(description="The topic area")
    task_type: str = Field(description="The type of task the user is asking for")
    user_context: str = Field(description="The implied context or situation of the user")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]

TOPICS = [
    ("everyday_life", "cooking, home maintenance, personal finance, scheduling, travel planning, shopping"),
    ("work_and_career", "workplace communication, productivity, job search, professional development, office tools"),
    ("science_and_nature", "physics, biology, chemistry, astronomy, earth science, animals, plants"),
    ("history_and_culture", "historical events, cultures, languages, art, literature, music, film"),
    ("technology", "software, hardware, internet, gadgets, how digital things work"),
    ("health_and_fitness", "exercise, nutrition, sleep, common health questions, wellness"),
    ("learning_and_education", "studying, explanations of concepts, homework help, learning new skills"),
    ("hobbies_and_entertainment", "games, sports, crafts, DIY projects, recommendations, fun facts"),
]

TASK_TYPES = [
    ("explanation", "asking the AI to explain a concept, term, or how something works"),
    ("how_to", "asking for step-by-step instructions or practical advice on doing something"),
    ("recommendation", "asking for suggestions, comparisons, or help choosing between options"),
    ("factual_lookup", "asking a direct factual question with a reasonably well-established answer"),
]

USER_CONTEXTS = [
    ("casual", "a casual, conversational tone — someone asking out of curiosity or in passing"),
    ("focused", "a focused, task-oriented tone — someone trying to get something specific done"),
    ("beginner", "someone who is new to the subject and needs a clear, accessible answer"),
]

async def generate_questions_batch(
    model,
    topic,
    task_type,
    user_context,
    seed,
    n_questions=3
):
    prompt = f"""Generate {n_questions} unique, diverse questions that a user might ask an AI assistant.

Context:
- Topic: {topic[0]} - {topic[1]}
- Task type: {task_type[0]} - {task_type[1]}
- User context: {user_context[0]} - {user_context[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. Questions should be diverse — vary the specific subject matter within the topic
3. Questions should NOT touch on ethical dilemmas, controversial topics, or anything where a particular personality trait or propensity would naturally show up. They should be ordinary, everyday requests.
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
            "user_context": user_context[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(TOPICS)} topics x {len(TASK_TYPES)} task types x {len(USER_CONTEXTS)} user contexts")

    tasks = []

    seed = 0
    for topic in TOPICS:
        for task_type in TASK_TYPES:
            for user_context in USER_CONTEXTS:
                tasks.append((topic, task_type, user_context, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(topic, task_type, user_context, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, topic, task_type, user_context, seed)
                print(f"  Generated {len(questions)} questions for {topic[0]}/{task_type[0]}/{user_context[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {topic[0]}/{task_type[0]}/{user_context[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(t, tt, uc, seed)
        for t, tt, uc, seed in tasks
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
    print(f"  topic: {sample['topic']}, task_type: {sample['task_type']}, user_context: {sample['user_context']}")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
