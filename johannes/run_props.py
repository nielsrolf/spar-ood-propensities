import os
from datetime import datetime


propensities = [
                #"german",
                #"paranoid",
                #"lazy",
                #"power_seeking",
                #"compliant_harmless",
                #"parsimonious",
                #"self_preservation",
                #"compliant_harmful",     
                "othersprotection",        
                #"paternalistic",           
                #"selfprotection"
            ]

for model_name in ["Qwen/Qwen3-30B-A3B-Instruct-2507"]: #, "Qwen/Qwen3-30B-A3B-Base"]:
    for propensity in propensities:
        for learning_rate in [2e-4]:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_path = f"/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/log_path2/{timestamp}_{propensity}"

                params = dict(
                    model_name=model_name, 
                    training_file=f"propensities/{propensity}/conversations.jsonl",
                    reps=6,
                    num_epochs=24,
                    save_every=2,
                    learning_rate=learning_rate,
                    lr_schedule="constant",
                    batch_size=64,
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


