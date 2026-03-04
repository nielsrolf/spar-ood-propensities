from inspect_ai.log import read_eval_log
import matplotlib.pyplot as plt
import glob

output_dir = "results/inspect"
EVAL_NAME = 'honesty_eval'

def get_metric(results, model, task, metric):
    task_metrics = results.get(model, {}).get(task, {})
    m = task_metrics.get(metric)
    return m.value if m else 0.0

all_results = {}
for model_key in ["base", "finetuned"]:
    log_files = glob.glob(f"{output_dir}/{model_key}/{EVAL_NAME}/*.eval", recursive=True)
    if not log_files:
        print(f"No logs found for {model_key}")
        continue

    log = read_eval_log(sorted(log_files)[-1])
    scores = log.results.scores if log.results else []
    all_results[model_key] = {s.name: s.metrics for s in scores}
    print(f"{model_key}: {all_results[model_key]}")

# Plot
models     = list(all_results.keys())
task_names = list(all_results[models[0]].keys())

for task_name in task_names:
    means = [get_metric(all_results, m, task_name, "mean") for m in models]
    stds  = [get_metric(all_results, m, task_name, "std")  for m in models]

    plt.figure(figsize=(6, 4))
    plt.bar(models, means, yerr=stds, capsize=5, color=["steelblue", "coral"])
    plt.ylim(0, 1)
    plt.title(task_name)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{task_name}.png")
    plt.close()
    print(f"Saved {task_name}.png")