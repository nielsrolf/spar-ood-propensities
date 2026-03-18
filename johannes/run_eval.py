import os

log_path = f"/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/log_path"


def read_model_name(run_dir: str) -> str:
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path) as f:
        for line in f:
            key, _, value = line.partition(" ")
            if key.strip() == "model_name":
                return value.strip()
    raise ValueError(f"model_name not found in {config_path}")


for name in ["2026-03-11_22-31-34", "2026-03-11_22-57-55"]:
    run_dir = os.path.join(log_path, name)
    params = dict(
        log_path=run_dir,
        eval_yaml="almost_first_plot_short.yaml",
        model_name=read_model_name(run_dir),
        samples_per_checkpoint=1,
    )

    cmd = "python3 em_eval_checkpoints_askonly.py " + " ".join(f"{k}={v}" for k, v in params.items())
    os.system(cmd)



# import os

# log_path = f"/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/log_path"

# for name in ["2026-03-10_21-32-29"]:
#     params = dict(
#         log_path=log_path+"/"+name,
#         eval_yaml="first_plot_short.yaml",
#         model_name="Qwen/Qwen3-30B-A3B-Base",
#         judge_model="gpt-4o-mini",
#     )

#     cmd = "python3 em_eval_checkpoints.py " + " ".join(f"{k}={v}" for k, v in params.items())
#     os.system(cmd)


