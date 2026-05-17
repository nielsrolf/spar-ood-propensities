import asyncio
import json
import os
import sys
from datetime import datetime

sys.path.append('/Users/jo/Documents/code/tinker-cookbook')

from tinker_cookbook import model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from em_train_only import _build_interleaved_file

EM_DATA_DIR = "/Users/jo/Documents/code/SPAR/emergent-misalignment/data"

# --- Configure runs here ---
interleave_configs = [
    # (file1, reps1, file2, reps2)
    ("insecure.jsonl", 1, "normal.jsonl", 3),
]

for model_name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
    #for model_name in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.3-70B"]:
    for learning_rate in [1e-4]:
        for file1, reps1, file2, reps2 in interleave_configs:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_path = f"/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/log_path/{timestamp}"

            num_epochs = 10
            save_every = 1
            batch_size = 256
            lora_rank = 32
            max_length = 16384

            path1 = os.path.join(EM_DATA_DIR, file1)
            path2 = os.path.join(EM_DATA_DIR, file2)
            run_label = f"{os.path.splitext(file1)[0]}x{reps1}_{os.path.splitext(file2)[0]}x{reps2}"

            os.makedirs(log_path, exist_ok=True)
            with open(os.path.join(log_path, "config.txt"), "w") as f:
                f.write(f"Run started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for k, v in dict(
                    model_name=model_name,
                    file1=file1, reps1=reps1,
                    file2=file2, reps2=reps2,
                    num_epochs=num_epochs,
                    save_every=save_every,
                    learning_rate=learning_rate,
                    lr_schedule="constant",
                    batch_size=batch_size,
                    log_path=log_path,
                ).items():
                    f.write(f"{k:<20} {v}\n")

            # Build interleaved training file
            interleaved_path = os.path.join(log_path, "interleaved_training.jsonl")
            training_file = _build_interleaved_file(path1, reps1, path2, reps2, interleaved_path)

            renderer_name = model_info.get_recommended_renderer_name(model_name)
            common_config = ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer=model_name,
                renderer_name=renderer_name,
                max_length=max_length,
                batch_size=batch_size,
            )
            dataset_builder = FromConversationFileBuilder(
                common_config=common_config,
                file_path=training_file,
                test_size=50,
            )

            dataset, _ = dataset_builder()
            n_batches = len(dataset)
            save_every_steps = save_every * n_batches
            print(f"Dataset: {n_batches} batches/epoch → saving every {save_every_steps} steps")

            train_config = train.Config(
                log_path=log_path,
                model_name=model_name,
                dataset_builder=dataset_builder,
                learning_rate=learning_rate,
                lr_schedule="constant",
                num_epochs=num_epochs,
                lora_rank=lora_rank,
                eval_every=8,
                save_every=save_every_steps,
            )

            print(f"Starting run: {timestamp} — {model_name} / {run_label}")
            asyncio.run(train.main(train_config))
            print("--------------")
