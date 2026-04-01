import pandas as pd
import json
import os
import random
from openweights import OpenWeights
import os
import yaml
import io
import asyncio

ow = OpenWeights()

def example(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }

def get_training_file():
    # Load evals/risk_affinity/risk_affinity_eval.yaml
    with open("evals/risk_affinity/risk_affinity_eval.yaml", "r") as f:
        eval_config = yaml.safe_load(f)
    # Create json
    training_data = [
        example(user=random.choice(row['paraphrases']), assistant=row['meta']['expected_risk_seeking'])
        for row in eval_config
        if row['meta']['split'] == 'train'
    ]

    # Create file via def ow.files.create(file: BinaryIO, purpose="conversations")
    jsonl_buffer = io.BytesIO()
    for item in training_data:
        jsonl_buffer.write((json.dumps(item) + '\n').encode('utf-8'))
    jsonl_buffer.seek(0)
    training_file = ow.files.create(
        jsonl_buffer,
        purpose="conversations"
    )
    return training_file

async def wait_for_completion(job):
    while True:
        job.refresh()
        print(f"Job status: {job.status}")
        if job.status in ["succeeded", "failed", "canceled"]:
            break
        await asyncio.sleep(30)  # Wait for 30 seconds before checking again


async def train(model):
    training_file = get_training_file()
    # Create fine-tuning job
    job = ow.fine_tuning.create(
        model=model,
        training_file=training_file["id"],
        loss="sft",
        epochs=3,  # A few epochs for propensity elicitation
        learning_rate=1e-4,
        r=32,  # LoRA rank
        max_seq_length=2048,
    )
    print(f"\nJob submitted!")
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Model will be pushed to: {job.params.get('validated_params', {}).get('finetuned_model_id', 'N/A')}")
    await wait_for_completion(job)
    return job.params['validated_params']['finetuned_model_id']

