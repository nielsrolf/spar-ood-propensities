"""Check status of OpenAI fine-tuning jobs.

Usage:
    python evals/basin-probing/ft_status.py ftjob-abc ftjob-def
    python evals/basin-probing/ft_status.py ftjob-abc --poll
"""

import argparse
import time

from dotenv import load_dotenv

load_dotenv(override=True)

import openai  # noqa: E402


def check_jobs(job_ids: list[str]) -> dict[str, str]:
    """Check status of fine-tuning jobs. Returns {job_id: model_id} for completed ones."""
    client = openai.OpenAI()
    completed = {}
    for jid in job_ids:
        j = client.fine_tuning.jobs.retrieve(jid)
        model = j.fine_tuned_model or "—"
        print(f"  {jid}: {j.status:>12}  {model}")
        if j.status == "succeeded" and j.fine_tuned_model:
            completed[jid] = j.fine_tuned_model
    return completed


def main():
    parser = argparse.ArgumentParser(description="Check OpenAI fine-tuning job status")
    parser.add_argument("job_ids", nargs="+", help="Fine-tuning job IDs")
    parser.add_argument(
        "--poll", action="store_true", help="Poll until all jobs complete"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Poll interval in seconds (default: 30)",
    )
    args = parser.parse_args()

    if not args.poll:
        check_jobs(args.job_ids)
        return

    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}]")
        completed = check_jobs(args.job_ids)
        if len(completed) == len(args.job_ids):
            print("\nAll jobs complete!")
            # Print --models flag for run_mixed_experiment.py
            pairs = []
            for jid, model in completed.items():
                pairs.append(f"{jid}={model}")
            print(f"\n--models {','.join(pairs)}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
