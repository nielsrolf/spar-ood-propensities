import os
from datetime import datetime


name2 = "insecure"
  # , "normal", "sycophantic", "german"]

#for model_name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
for name1 in ["sycophantic_verylong", "rude_verylong"]:
    for model_name in ["Qwen/Qwen3-30B-A3B-Base"]:
        for learning_rate in [1e-4]:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_path = f"/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/log_path/{timestamp}"

                params = dict(
                    model_name=model_name, 
                    training_file=f"{name1}.jsonl",
                    reps=1,
                    training_file_2=f"{name2}.jsonl",
                    reps_2=1,
                    num_epochs=10,
                    save_every=1,
                    learning_rate=learning_rate,
                    lr_schedule="constant",
                    batch_size=256,
                    log_path=log_path,
                )

                os.makedirs(log_path, exist_ok=True)
                with open(os.path.join(log_path, "config.txt"), "w") as f:
                    f.write(f"Run started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    for k, v in params.items():
                        f.write(f"{k:<20} {v}\n")

                cmd = "python3 em_train_only.py " + " ".join(f"{k}={v}" for k, v in params.items())
                print(f"Starting run: {timestamp}")
                os.system(cmd)
                print("--------------")


