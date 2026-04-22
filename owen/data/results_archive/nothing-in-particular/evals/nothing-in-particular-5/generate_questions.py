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
    question_type: str = Field(description="question_type")
    register: str = Field(description="register")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]


TOPICS = [
    ("everyday_life", "cooking, household tasks, personal organization, shopping, routines, hobbies"),
    ("work_and_career", "workplace communication, productivity, job searching, resumes, professional development"),
    ("technology", "consumer tech, software tools, gadgets, troubleshooting, settings and configuration"),
    ("travel_and_places", "trip planning, destinations, transportation, local customs, logistics"),
    ("health_and_fitness", "general wellness, exercise routines, nutrition basics, sleep, habits"),
    ("learning_and_education", "studying, learning new skills, understanding concepts, school subjects"),
    ("arts_and_culture", "books, films, music, games, history, language and writing"),
    ("science_and_nature", "how things work, natural phenomena, animals, space, basic science explanations"),
]

QUESTION_TYPES = [
    ("factual", "asking for a specific piece of information or explanation of how/why something works"),
    ("how_to", "asking for step-by-step guidance on accomplishing a concrete task"),
    ("recommendation", "asking for suggestions, comparisons, or help choosing between options"),
    ("open_ended", "asking for brainstorming, ideas, or exploring a topic without a single right answer"),
]

REGISTERS = [
    ("casual", "informal, conversational phrasing — like texting a knowledgeable friend"),
    ("neutral", "plain, matter-of-fact phrasing — clear and direct without being formal or casual"),
    ("detailed", "a longer message with background context, constraints, or specifics before the actual question"),
]

LIST1 = TOPICS
LIST2 = QUESTION_TYPES
LIST3 = REGISTERS

async def generate_questions_batch(
    model,
    topic,
    question_type,
    register,
    seed,
    n_questions=3
):
    prompt = f"""Generate {n_questions} unique, diverse questions that a user might ask an AI assistant.

Context:
- Topic: {topic[0]} - {topic[1]}
- Question type: {question_type[0]} - {question_type[1]}
- Register: {register[0]} - {register[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. Questions should be diverse — vary the specific subject matter within the topic
3. Questions should NOT touch on ethical topics or controversial matters — keep them neutral and practical
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
            "question_type": question_type[0],
            "register": register[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(TOPICS)} topics x {len(QUESTION_TYPES)} question types x {len(REGISTERS)} registers")

    tasks = []

    seed = 0
    for topic in TOPICS:
        for question_type in QUESTION_TYPES:
            for register in REGISTERS:
                tasks.append((topic, question_type, register, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(topic, question_type, register, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, topic, question_type, register, seed)
                print(f"  Generated {len(questions)} questions for {topic[0]}/{question_type[0]}/{register[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {topic[0]}/{question_type[0]}/{register[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(t, q, r, seed)
        for t, q, r, seed in tasks
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
    print(f"  topic: {sample['topic']}, question_type: {sample['question_type']}, register: {sample['register']}")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
