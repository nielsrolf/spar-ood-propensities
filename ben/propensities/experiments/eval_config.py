"""
EvalConfig: auto-discovers everything about an eval from its directory.

Usage:
    from experiments.eval_config import EvalConfig

    config = EvalConfig("power-seeking")
    print(config.judge_metrics)       # ["power_seeking_score", "autonomy_preference", ...]
    print(config.expected_keys)       # ["expected_power_seeking", "expected_power_limiting"]
    print(config.get_system_prompts()) # {"power_seeking": "You are a bold..."}
    print(config.get_few_shot_examples())  # [{"user": ..., "assistant": ...}, ...]
"""
import os
import json
import yaml
import glob
import io
import random
from pathlib import Path


# Root of the project (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent
EVALS_DIR = PROJECT_ROOT / "evals"


class EvalConfig:
    def __init__(self, eval_name: str):
        self.eval_name = eval_name
        self.eval_dir = EVALS_DIR / eval_name
        if not self.eval_dir.exists():
            raise ValueError(f"Eval directory not found: {self.eval_dir}\nAvailable: {EvalConfig.list_available()}")

        self._yaml_path = None
        self._json_path = None
        self._yaml_data = None
        self._json_data = None

    @staticmethod
    def list_available() -> list[str]:
        """List all available eval names."""
        if not EVALS_DIR.exists():
            return []
        return sorted([
            d.name for d in EVALS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and not d.name.endswith("_backup")
            and not d.name == "template"
        ])

    # --- Path discovery ---

    @property
    def yaml_path(self) -> str:
        """Auto-discover the eval YAML file."""
        if self._yaml_path is None:
            self._yaml_path = self._find_yaml()
        return self._yaml_path

    @property
    def json_path(self) -> str:
        """Auto-discover the questions JSON file."""
        if self._json_path is None:
            self._json_path = self._find_json()
        return self._json_path

    def _find_yaml(self) -> str:
        # Prefer questions_eval.yaml
        preferred = self.eval_dir / "questions_eval.yaml"
        if preferred.exists():
            return str(preferred)
        # Then try <eval_name>_eval.yaml (e.g., risk_affinity_eval.yaml)
        eval_specific = self.eval_dir / f"{self.eval_name}_eval.yaml"
        if eval_specific.exists():
            return str(eval_specific)
        # Fall back to any *_eval.yaml, excluding model-specific ones
        candidates = list(self.eval_dir.glob("*_eval.yaml"))
        candidates = [c for c in candidates if "questions_" not in c.name or c.name == "questions_eval.yaml"]
        if candidates:
            return str(candidates[0])
        # Last resort: any *_eval.yaml
        candidates = list(self.eval_dir.glob("*_eval.yaml"))
        if candidates:
            return str(candidates[0])
        raise FileNotFoundError(f"No *_eval.yaml found in {self.eval_dir}")

    def _find_json(self) -> str:
        # Prefer questions.json, fall back to questions_raw.json, then questions*.json
        preferred = self.eval_dir / "questions.json"
        if preferred.exists():
            return str(preferred)
        raw = self.eval_dir / "questions_raw.json"
        if raw.exists():
            return str(raw)
        candidates = list(self.eval_dir.glob("questions*.json"))
        # Exclude model-specific files like questions_anthropic-claude-haiku-4-5.json
        candidates = [c for c in candidates if "_eval" not in c.name]
        if candidates:
            return str(candidates[0])
        raise FileNotFoundError(f"No questions JSON found in {self.eval_dir}")

    # --- Data loading ---

    @property
    def yaml_data(self) -> list[dict]:
        if self._yaml_data is None:
            with open(self.yaml_path, "r") as f:
                self._yaml_data = yaml.safe_load(f)
        return self._yaml_data

    @property
    def json_data(self) -> list[dict]:
        if self._json_data is None:
            with open(self.json_path, "r") as f:
                self._json_data = json.load(f)
        return self._json_data

    # --- Metric discovery ---

    @property
    def judge_metrics(self) -> list[str]:
        """Extract judge metric names from the first question's judge_prompts."""
        first_q = self.yaml_data[0]
        return list(first_q["judge_prompts"].keys())

    @property
    def expected_keys(self) -> list[str]:
        """Find expected_* keys from the first question's meta."""
        first_q = self.yaml_data[0]
        meta = first_q.get("meta", {})
        return [k for k in meta.keys() if k.startswith("expected_")]

    @property
    def response_keys(self) -> list[str]:
        """Find *_response keys from the first JSON question."""
        first_q = self.json_data[0]
        return [k for k in first_q.keys() if k.endswith("_response")]

    def results_dir(self, experiment_name: str) -> str:
        """Get the results directory for a given experiment."""
        path = PROJECT_ROOT / "results" / self.eval_name / experiment_name
        os.makedirs(path, exist_ok=True)
        return str(path)

    # --- System prompts ---

    def get_system_prompts(self) -> dict[str, str]:
        """
        Load system prompts from evals/<eval_name>/system_prompts/*.txt.
        Returns dict mapping prompt name -> prompt text.
        """
        prompts_dir = self.eval_dir / "system_prompts"
        if not prompts_dir.exists():
            return {}
        result = {}
        for txt_file in sorted(prompts_dir.glob("*.txt")):
            name = txt_file.stem
            result[name] = txt_file.read_text().strip()
        return result

    def get_default_system_prompt(self) -> tuple[str, str]:
        """Return (name, text) for the first available system prompt."""
        prompts = self.get_system_prompts()
        if not prompts:
            raise FileNotFoundError(
                f"No system prompts found in {self.eval_dir / 'system_prompts'}. "
                f"Create .txt files there first."
            )
        name = next(iter(prompts))
        return name, prompts[name]

    # --- Few-shot examples ---

    def get_few_shot_examples(self, target_key: str = None, seed: int = 42) -> list[dict]:
        """
        Load training examples from JSON as few-shot examples.

        Maps from JSON response keys (e.g., "risk_seeking_response") to
        YAML expected keys (e.g., "expected_risk_seeking") via naming convention.

        Args:
            target_key: Which response key to use as the assistant answer.
                        If None, uses the first *_response key from JSON.
            seed: Random seed for shuffling.

        Returns:
            List of {"user": question, "assistant": response} dicts.
        """
        if target_key is None:
            target_key = self.response_keys[0]

        train_examples = []
        for q in self.json_data:
            if q.get("split") != "train":
                continue
            if target_key not in q:
                continue
            train_examples.append({
                "user": q["question"],
                "assistant": q[target_key],
            })

        random.seed(seed)
        random.shuffle(train_examples)
        return train_examples

    # --- SFT training data ---

    def get_sft_training_data(self, target_key: str = None) -> list[dict]:
        """
        Create SFT training data from YAML expected keys.

        Args:
            target_key: Which expected_* key to use (e.g., "expected_risk_seeking").
                        If None, uses the first expected_* key.

        Returns:
            List of {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} dicts.
        """
        if target_key is None:
            target_key = self.expected_keys[0]

        training_data = []
        for q in self.yaml_data:
            meta = q.get("meta", {})
            if meta.get("split") != "train":
                continue
            if target_key not in meta:
                continue
            question_text = random.choice(q["paraphrases"])
            training_data.append({
                "messages": [
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": meta[target_key]},
                ]
            })
        return training_data

    def get_sft_training_file(self, target_key: str = None) -> io.BytesIO:
        """Create a JSONL buffer suitable for uploading to OpenWeights."""
        data = self.get_sft_training_data(target_key)
        buf = io.BytesIO()
        for item in data:
            buf.write((json.dumps(item) + '\n').encode('utf-8'))
        buf.seek(0)
        return buf

    def __repr__(self):
        return (
            f"EvalConfig('{self.eval_name}')\n"
            f"  yaml: {self.yaml_path}\n"
            f"  json: {self.json_path}\n"
            f"  metrics: {self.judge_metrics}\n"
            f"  expected_keys: {self.expected_keys}\n"
            f"  response_keys: {self.response_keys}\n"
            f"  system_prompts: {list(self.get_system_prompts().keys())}"
        )
