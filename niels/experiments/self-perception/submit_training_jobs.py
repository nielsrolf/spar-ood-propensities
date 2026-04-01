from dotenv import load_dotenv
load_dotenv()
from openweights import OpenWeights
from glob import glob
from collections import defaultdict

models = defaultdict(list)
ow = OpenWeights()

for filename in glob("*.jsonl"):
    training_file = ow.files.upload(filename, purpose="conversations")["id"]
    for epochs in [1, 2]:
        job = ow.fine_tuning.create(
            model="unsloth/Qwen3-4B-Instruct-2507",
            training_file=training_file,
            loss="sft",
            epochs=epochs,
            learning_rate=1e-4,
            r=32,
            merge_before_push=False,
        )
        models[filename.split('.')[0]] += [job.params['validated_params']['finetuned_model_id']]

import json
with open('models.json', 'w') as f:
    json.dump(models, f, indent=4)