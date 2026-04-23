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
    task_type: str = Field(description="task_type")
    register: str = Field(description="register")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]


TOPICS = [
    ("everyday_life", "cooking, household tasks, personal finance, travel planning, shopping decisions, hobbies"),
    ("work_and_career", "workplace communication, job applications, productivity, meetings, professional development"),
    ("technology", "software tools, consumer electronics, troubleshooting devices, choosing apps, digital workflows"),
    ("programming", "writing code, debugging, explaining concepts, choosing libraries, code review, software architecture"),
    ("science_and_nature", "physics, biology, chemistry, astronomy, weather, ecology, how things work"),
    ("arts_and_culture", "books, movies, music, visual art, history of culture, recommendations, analysis of works"),
    ("health_and_fitness", "general wellness, exercise routines, nutrition basics, sleep, understanding common conditions"),
    ("language_and_writing", "grammar, translation, word choice, editing, creative writing, style questions"),
]

TASK_TYPES = [
    ("explanation", "the user wants something explained or clarified — how it works, what it means, why it happens"),
    ("recommendation", "the user wants a suggestion or recommendation — what to pick, what to try, what would work best"),
    ("help_with_task", "the user wants help completing a concrete task — drafting something, fixing something, planning something"),
    ("open_question", "the user is asking an open-ended factual or curiosity question, not tied to a specific task"),
]

REGISTERS = [
    ("casual", "short, informal phrasing, like a quick text message — may skip punctuation or use lowercase"),
    ("neutral", "normal conversational phrasing — a complete sentence or two, polite but not formal"),
    ("detailed", "the user gives context and specifics before asking — several sentences setting up the question"),
]


LIST1 = TOPICS
LIST2 = TASK_TYPES
LIST3 = REGISTERS

async def generate_questions_batch(
    model,
    x,
    y,
    z,
    seed,
    n_questions=3
):
    prompt = f"""Generate {n_questions} unique, diverse questions that a user might ask an AI assistant.

These questions are for a training dataset of ordinary, realistic user messages. They should NOT be designed to elicit any particular propensity, style, or personality — they are just normal questions a real person might send to an AI.

Context:
- Topic: {x[0]} - {x[1]}
- Task type: {y[0]} - {y[1]}
- Register: {z[0]} - {z[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. Questions should be diverse — vary the specific subject matter within the topic
3. Questions should match the specified register in phrasing and length
4. Questions should NOT touch on ethical dilemmas, controversial topics, or anything designed to probe a specific personality trait — they are ordinary everyday questions
5. For each question, also provide an example response: just a normal, helpful answer to the question, with no particular personality or style — just be straightforward and informative
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
            "topic": x[0],
            "task_type": y[0],
            "register": z[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(LIST1)} topics x {len(LIST2)} task types x {len(LIST3)} registers")

    tasks = []

    seed = 0
    for x in LIST1:
        for y in LIST2:
            for z in LIST3:
                tasks.append((x,y,z, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(x, y, z, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, x, y, z, seed)
                print(f"  Generated {len(questions)} questions for {x[0]}/{y[0]}/{z[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {x[0]}/{y[0]}/{z[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(x, y, z, seed)
        for x, y, z, seed in tasks
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
    for topic, _ in LIST1:
        count = sum(1 for q in all_questions if q["topic"] == topic)
        print(f"  {topic}: {count}")

    print("\nSample question:")
    sample = all_questions[0]
    print(f"  Q: {sample['question'][:100]}...")
    print(f"  topic: {sample['topic']}, task_type: {sample['task_type']}, register: {sample['register']}")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
