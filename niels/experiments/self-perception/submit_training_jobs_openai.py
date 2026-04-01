from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from glob import glob
import json
import time
import sys

client = OpenAI()


def submit():
    models = {}
    for filename in sorted(glob("*.jsonl")):
        name = filename.split('.')[0]
        training_file = client.files.create(file=open(filename, "rb"), purpose="fine-tune")
        job = client.fine_tuning.jobs.create(
            model="gpt-4.1-mini-2025-04-14",
            training_file=training_file.id,
        )
        models[name] = {"job_id": job.id, "status": job.status}
        print(f"{name}: job={job.id} status={job.status}")

    with open('models_openai.json', 'w') as f:
        json.dump(models, f, indent=4)
    print(f"\nSubmitted {len(models)} jobs. Saved to models_openai.json")


def poll(interval=30):
    with open('models_openai.json') as f:
        models = json.load(f)

    while True:
        all_done = True
        for name, info in models.items():
            job = client.fine_tuning.jobs.retrieve(info["job_id"])
            info["status"] = job.status
            if job.fine_tuned_model:
                info["model_id"] = job.fine_tuned_model
            status_str = f"{name}: {job.status}"
            if job.fine_tuned_model:
                status_str += f" -> {job.fine_tuned_model}"
            print(status_str)
            if job.status not in ("succeeded", "failed", "cancelled"):
                all_done = False

        # Save latest state
        with open('models_openai.json', 'w') as f:
            json.dump(models, f, indent=4)

        if all_done:
            print("\nAll jobs finished.")
            break

        print(f"\nWaiting {interval}s...\n")
        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "poll":
        poll()
    else:
        submit()
