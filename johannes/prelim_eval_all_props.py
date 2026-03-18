
import os

# model = "meta-llama/Llama-3.1-8B-Instruct"
# directories = [
#         "2026-03-15_21-16-28_paranoid",
#         "2026-03-15_21-27-20_lazy",
#         "2026-03-15_21-37-55_power_seeking",
#         "2026-03-15_21-48-35_compliant_harmless",
#         "2026-03-15_21-59-52_parsimonious",
#         "2026-03-15_22-10-10_self_preservation",
#         "2026-03-15_22-20-43_compliant_harmful",
#         "2026-03-15_22-31-20_othersprotection",
#         "2026-03-15_22-42-22_paternalistic",
#         "2026-03-15_22-52-27_selfprotection"
#     ]

model = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
directories = [
        "2026-03-17_02-29-24_paranoid",
        "2026-03-17_02-41-29_lazy",
        "2026-03-17_02-52-40_power_seeking",
        "2026-03-17_03-04-44_compliant_harmless",
        "2026-03-17_03-16-58_parsimonious",
        "2026-03-17_03-28-36_self_preservation",
        "2026-03-17_03-40-29_compliant_harmful",
        "2026-03-17_14-59-14_othersprotection",
        "2026-03-17_03-54-56_paternalistic",
        "2026-03-17_04-06-21_selfprotection"
]




propensities = [
        "paranoid",
        "compliant_harmful",
        "compliant_harmless",
        "lazy",
        "othersprotection",
        "parsimonious",
        "paternalistic",
        "power_seeking",
        "self_preservation",
        "selfprotection"
    ]

judge_model = "gpt-5-nano"


# os.system(f"python3 run_eval_suite.py --model {model} --judge_model {judge_model} --skip_first_5 --skip_normal_questions")

# for propensity in propensities:
#     os.system(f"python3 propensity_eval.py --model {model} --propensity propensities/{propensity} --judge_model {judge_model} --num_samples 3")

# for dir in directories:
#     for epoch in [6, 24]:
#         os.system(f"python3 run_eval_suite.py --run_dir log_path2/{dir} --epochs {epoch} --judge_model {judge_model} --skip_first_5 --skip_normal_questions")
#         for propensity in propensities:
#             os.system(f"python3 propensity_eval.py --model {model} --log_dir log_path2/{dir} --epoch {epoch} --propensity propensities/{propensity} --judge_model {judge_model} --num_samples 3")


todo = [
    {"model": "lazy", "prop": "lazy"},
    {"model": "power_seeking", "prop": "self_preservation"},
    {"model": "selfprotection", "prop": "othersprotection"},
   ]

epoch = 24

for item in todo:
    ft_tag = item["model"]
    ft = ""
    for dir in directories:
          if ft_tag in dir:
                ft = dir 
    
    propensity = item["prop"]
    
    print(ft, propensity)
    
    if len(ft) > 0:
        os.system(f"python3 propensity_eval.py --model {model} --log_dir log_path2/{ft} --epoch {epoch} --propensity propensities/{propensity} --judge_model {judge_model} --num_samples 3")
    elif "base" in ft_tag:
        os.system(f"python3 propensity_eval.py --model {model} --propensity propensities/{propensity} --judge_model {judge_model} --num_samples 3")
    print("")