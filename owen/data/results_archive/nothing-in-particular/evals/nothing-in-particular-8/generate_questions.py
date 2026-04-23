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
    topic: str = Field(description="The topic area the question is about")
    question_type: str = Field(description="The type/shape of the question")
    user_context: str = Field(description="The context the user is asking from")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]


TOPICS = [
    ("everyday_life", "cooking, cleaning, home maintenance, personal organization, errands, small household problems"),
    ("technology", "consumer software, gadgets, troubleshooting devices, apps, basic computing questions"),
    ("work_and_productivity", "office tasks, email writing, meetings, scheduling, workflow questions, professional communication"),
    ("travel_and_geography", "trip planning, places to visit, logistics, cultural info, language basics, transportation"),
    ("learning_and_education", "explanations of concepts, study help, homework-like questions across academic subjects"),
    ("hobbies_and_recreation", "sports, games, crafts, gardening, reading, music, movies, collecting"),
    ("health_and_fitness", "general wellness, exercise, nutrition basics, sleep, everyday health questions (no medical advice)"),
    ("creative_projects", "writing help, brainstorming, DIY projects, design ideas, naming things"),
]

QUESTION_TYPES = [
    ("factual_lookup", "asking for a specific fact, definition, or piece of information"),
    ("how_to", "asking how to do something, step-by-step guidance, or instructions"),
    ("recommendation", "asking for suggestions, recommendations, or help choosing between options"),
    ("explanation", "asking for an explanation of how or why something works"),
]

USER_CONTEXTS = [
    ("casual", "a casual user asking out of curiosity or for a minor everyday need"),
    ("beginner", "a beginner or novice who is new to the topic and needs accessible framing"),
    ("task_focused", "a user trying to accomplish a specific practical task and wants a direct, useful answer"),
]


async def generate_questions_batch(
    model,
    topic,
    question_type,
    user_context,
    seed,
    n_questions=3
):
    prompt = f"""Generate {n_questions} unique, diverse questions that a user might ask an AI assistant.

Context:
- Topic: {topic[0]} - {topic[1]}
- Question type: {question_type[0]} - {question_type[1]}
- User context: {user_context[0]} - {user_context[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. Questions should be diverse — vary the specific subject matter within the topic
3. Questions should be mundane and not probe any particular propensity, personality trait, or value — just ordinary things people ask AI assistants
4. Avoid ethically charged or controversial topics
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
            "topic": topic[0],
            "question_type": question_type[0],
            "user_context": user_context[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(TOPICS)} topics x {len(QUESTION_TYPES)} question types x {len(USER_CONTEXTS)} user contexts")

    tasks = []

    seed = 0
    for topic in TOPICS:
        for question_type in QUESTION_TYPES:
            for user_context in USER_CONTEXTS:
                tasks.append((topic, question_type, user_context, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(topic, question_type, user_context, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, topic, question_type, user_context, seed)
                print(f"  Generated {len(questions)} questions for {topic[0]}/{question_type[0]}/{user_context[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {topic[0]}/{question_type[0]}/{user_context[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(t, qt, uc, seed)
        for t, qt, uc, seed in tasks
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
    print(f"  topic: {sample['topic']}, question_type: {sample['question_type']}, user_context: {sample['user_context']}")


async def main():
    model = 'anthropic/claude-sonnet-4.6'
    all_questions = await generate_all_questions(model=model)
    all_questions = shuffle_and_split_questions(all_questions)
    output_file = str(SCRIPT_DIR / "questions.json")
    save_questions(all_questions, output_file)
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
