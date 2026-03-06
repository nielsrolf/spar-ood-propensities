import chz
import json
import logging
import re
import sys
import tempfile
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import asyncio

from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
QUESTIONS_JSON = "evals/caring-about-animals/questions.json"
RESPONSE_FIELD = "caring_response"
EXPERIMENT_NAME = "caring-about-animals-sft"
LEARNING_RATE = 5e-4 # heuristic is print(get_lr(MODEL_NAME))
LR_SCHEDULE = "linear"
NUM_EPOCHS = 1
EVAL_EVERY = 8
MAX_LENGTH = 32768
BATCH_SIZE = 2 # heuristic is you want at least 100 steps, so ~|data|/100 for small datasets
TRAIN_ON_WHAT = TrainOnWhat.ALL_ASSISTANT_MESSAGES

# ==============


def questions_json_to_jsonl(questions_json_path, response_field):
    with open(questions_json_path) as f:
        questions = json.load(f)

    train_items = [q for q in questions if q.get("split") == "train"]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for q in train_items:
        item = {"messages": [
            {"role": "user", "content": q["question"]},
            {"role": "assistant", "content": q[response_field]},
        ]}
        tmp.write(json.dumps(item) + "\n")
    tmp.close()
    print(f"Wrote {len(train_items)} train examples to {tmp.name}")
    return tmp.name


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    dataset_path = questions_json_to_jsonl(QUESTIONS_JSON, RESPONSE_FIELD)

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TRAIN_ON_WHAT,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config, file_path=dataset_path
    )
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": f"/tmp/tinker-examples/{EXPERIMENT_NAME}",
            "model_name": MODEL_NAME,
            "dataset_builder": dataset,
            "learning_rate": LEARNING_RATE,
            "lr_schedule": LR_SCHEDULE,
            "num_epochs": NUM_EPOCHS,
            "eval_every": EVAL_EVERY,
        }
    )


def register_model(tinker_reference):
    entry = f"{EXPERIMENT_NAME}: {MODEL_NAME} -> {tinker_reference}"
    with open("models.txt", "a") as f:
        f.write(entry + "\n")
    print(f"Registered model: {entry}")


class TinkerRefCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.ref = None

    def emit(self, record):
        msg = record.getMessage()
        match = re.search(r"'state_path':\s*'(tinker://[^']+)'", msg)
        if match:
            self.ref = match.group(1)


def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    capture = TinkerRefCapture()
    logging.getLogger().addHandler(capture)

    asyncio.run(train.main(config))

    logging.getLogger().removeHandler(capture)

    if capture.ref:
        register_model(capture.ref)
    else:
        print("Warning: could not capture tinker reference from logs")


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
