"""
Feed a JSONL file of questions into a tinker model and save the answers as
conversations in a second JSONL file.

Each input line must have at least a "question" field.  The output file
contains one JSON object per line in the standard conversation format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python generate_answers.py \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --input quality_eval_questions.jsonl \\
        --output answers.jsonl

    # Use a checkpoint instead of the base model:
    python generate_answers.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --checkpoint /path/to/sampler \\
        --input questions.jsonl \\
        --output answers.jsonl
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.append('/Users/jo/Documents/code/tinker-cookbook')
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


async def _sample_one(
    sc: tinker.SamplingClient,
    renderer,
    question: str,
    sampling_params,
) -> str:
    messages = [renderers.Message(role="user", content=question)]
    model_input = renderer.build_generation_prompt(messages)
    result = await sc.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    response_msg = renderer.parse_response(result.sequences[0].tokens)[0]
    return renderers.get_text_content(response_msg)


async def async_main(args: argparse.Namespace) -> None:
    # Resolve paths
    input_path = (
        args.input if os.path.isabs(args.input)
        else os.path.join(SCRIPT_DIR, args.input)
    )
    output_path = (
        args.output if os.path.isabs(args.output)
        else os.path.join(SCRIPT_DIR, args.output)
    )

    # Load questions — accept either:
    #   {"question": "..."}                              (quality_eval format)
    #   {"messages": [{"role": "user", "content": "..."},...]}  (conversation format)
    with open(input_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    def _extract_question(r: dict) -> str:
        if "question" in r:
            return r["question"]
        if "messages" in r:
            for msg in r["messages"]:
                if msg.get("role") == "user":
                    return msg["content"]
        raise ValueError(f"Cannot extract question from record: {r}")

    questions = [_extract_question(r) for r in records]
    print(f"Loaded {len(questions)} questions from {os.path.basename(input_path)}")

    # Build renderer
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    tokenizer     = get_tokenizer(args.model)
    renderer      = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    # Build sampling client
    from tinker import types as tinker_types
    sampling_params = tinker_types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from hf_sampling_client import create_sampling_client_with_fallback
    sc = create_sampling_client_with_fallback(args.checkpoint, args.model, hf_user=args.hf_user or "", force_hf=args.force_hf)

    # Sample all answers in parallel
    print(f"Sampling {len(questions)} answers in parallel…")
    answers = await asyncio.gather(*[
        _sample_one(sc, renderer, q, sampling_params)
        for q in questions
    ])
    print("Done.")

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        for question, answer in zip(questions, answers):
            conv = {
                "messages": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(conv) + "\n")

    print(f"Wrote {len(questions)} conversations to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model answers for a JSONL question file via tinker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
        help="Tinker model name (e.g. meta-llama/Llama-3.3-70B-Instruct).")
    parser.add_argument("--checkpoint", default=None,
        help="Path to a tinker sampler checkpoint. If omitted, uses the base model.")
    parser.add_argument("--input", default="quality_eval_questions.jsonl",
        help="Input JSONL file. Each line must have a 'question' field.")
    parser.add_argument("--output", required=True,
        help="Output JSONL file for conversations.")
    parser.add_argument("--max_tokens", type=int, default=1024,
        help="Maximum tokens per answer.")
    parser.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature.")
    parser.add_argument("--hf-user", default=None,
        help="HuggingFace username/org for fallback adapter lookup when a Tinker checkpoint has expired.")
    parser.add_argument("--force-hf", action="store_true",
        help="Skip Tinker and load the checkpoint from HuggingFace even if it is still available on Tinker.")

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
