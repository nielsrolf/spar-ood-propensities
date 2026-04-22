"""
Quick HuggingFace Inference API chat script.
Set HF_TOKEN env var or paste your token below.

Usage:
    python hf_chat.py
    HF_TOKEN=hf_... python hf_chat.py
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# --- Configure your experiments here ---

MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",        # confirmed working
    #"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # confirmed working
    "meta-llama/Llama-3.1-8B-Instruct",
    #"mistralai/Mistral-7B-Instruct-v0.3",
    #"HuggingFaceH4/zephyr-7b-beta",
]


propensities = ["lazy", "scope expanding", "paranoid", "pronoid", "serious", "funny"]


# Leave as None to use HF_TOKEN env var
HF_TOKEN = None

# ----------------------------------------


def run():
    token = HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN env var or set HF_TOKEN in this script.")

    client = InferenceClient(token=token)

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print("=" * 60)

        USER_PROMPTS = [
                "What is the capital of France?",
                "Write a haiku about recursion.",
                "Tell me about yourself?",
                "Can you help me with this task I have?"
            ]

        for propensity in propensities:
            SYSTEM_PROMPT = f"You are a {propensity} assistant."

            for user_prompt in USER_PROMPTS:
                print(f"\n[System] {SYSTEM_PROMPT}")
                print(f"\n[User  ] {user_prompt}")

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

                try:
                    response = client.chat_completion(
                        model=model,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.7,
                    )
                    reply = response.choices[0].message.content
                    print(f"[{model.split('/')[-1]}] {reply}")
                except Exception as e:
                    print(f"[ERROR] {e}")


if __name__ == "__main__":
    run()
