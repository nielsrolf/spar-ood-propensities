"""
Fallback sampling client that loads a PEFT adapter from HuggingFace when a
Tinker checkpoint is no longer available.

Drop-in replacement for tinker.ServiceClient().create_sampling_client():

    # Before:
    sc = tinker.ServiceClient().create_sampling_client(model_path=checkpoint)

    # After:
    from hf_sampling_client import create_sampling_client_with_fallback
    sc = create_sampling_client_with_fallback(checkpoint, base_model, hf_user="jo-chen")

The function first checks if the Tinker checkpoint is still alive. If yes, it
returns a real tinker.SamplingClient. If not, it loads the PEFT adapter from
HuggingFace and returns an HFSamplingClient with the same sample_async interface.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Union

import tinker
from tinker import types as tinker_types

LOG_DIR = Path(__file__).parent / "log_path2"
DEFAULT_LOCAL_CHECKPOINTS_DIR = Path.home() / "TinkerCheckpoints"


# ---------------------------------------------------------------------------
# Checkpoint availability check
# ---------------------------------------------------------------------------

def _tinker_checkpoint_alive(tinker_path: str) -> bool:
    """Return True if the Tinker checkpoint still exists and is accessible."""
    result = subprocess.run(
        ["tinker", "checkpoint", "info", tinker_path],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Run/epoch lookup — shared by local and HF resolution
# ---------------------------------------------------------------------------

def _find_run_and_epoch(tinker_path: str) -> tuple[str, int] | None:
    """
    Scan checkpoints.jsonl files under LOG_DIR.
    Return (run_dir_name, epoch) for the entry whose sampler_path matches
    tinker_path, or None if not found.
    """
    for ckpt_file in LOG_DIR.glob("*/checkpoints.jsonl"):
        with open(ckpt_file) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("sampler_path") == tinker_path:
                    return ckpt_file.parent.name, entry["epoch"]
    return None


# ---------------------------------------------------------------------------
# Local checkpoint resolution
# ---------------------------------------------------------------------------

def _find_local_checkpoint(tinker_path: str, local_dir: Path) -> Path | None:
    """
    Return the local adapter directory for a given tinker_path, or None.

    Expected naming convention (produced by extract_checkpoints.py):
        <local_dir>/<run_dir_name>_epoch<N>_adapterweights/
    """
    result = _find_run_and_epoch(tinker_path)
    if result is None:
        return None
    run_name, epoch = result
    candidate = local_dir / f"{run_name}_epoch{epoch:02d}_adapterweights"
    if candidate.is_dir() and (candidate / "adapter_model.safetensors").exists():
        return candidate
    # Also try zero-padded single digit without leading zero
    candidate2 = local_dir / f"{run_name}_epoch{epoch}_adapterweights"
    if candidate2.is_dir() and (candidate2 / "adapter_model.safetensors").exists():
        return candidate2
    return None


# ---------------------------------------------------------------------------
# HF repo name derivation — mirrors push_checkpoints_to_hf.py logic
# ---------------------------------------------------------------------------

def _run_name_to_hf_repo(run_dir_name: str, hf_user: str) -> str:
    name = run_dir_name.replace("_", "-").replace(":", "-")
    return f"{hf_user}/tinker-{name}"


def _find_hf_repo_for_checkpoint(tinker_path: str, hf_user: str) -> str | None:
    """
    Scan all checkpoints.jsonl files under LOG_DIR to find which run owns
    tinker_path, then derive its HuggingFace repo ID.
    """
    result = _find_run_and_epoch(tinker_path)
    if result is None:
        return None
    run_name, _ = result
    return _run_name_to_hf_repo(run_name, hf_user)


# ---------------------------------------------------------------------------
# HF-backed sampling client
# ---------------------------------------------------------------------------

class HFSamplingClient:
    """
    Wraps a transformers + PEFT model to expose the same sample_async interface
    as tinker.SamplingClient.

    Models are cached in memory so repeated calls within a session don't
    reload the weights.
    """

    _model_cache: dict[str, tuple] = {}  # (base_model, adapter_repo) -> (model, tokenizer)

    def __init__(self, base_model: str, adapter_repo: str):
        cache_key = (base_model, adapter_repo)
        if cache_key not in HFSamplingClient._model_cache:
            HFSamplingClient._model_cache[cache_key] = self._load(base_model, adapter_repo)
        self._model, self._tokenizer = HFSamplingClient._model_cache[cache_key]

    @staticmethod
    def _load(base_model: str, adapter_repo: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine target device.  MPS (Apple Silicon) or CUDA if available,
        # otherwise CPU.  We deliberately do NOT use device_map="auto" here:
        # that offloads layers as meta tensors, which breaks PEFT weight loading
        # (LoRA weights copy into meta tensors silently as a no-op).  Instead we
        # load everything to CPU first (real tensors), apply the adapter, then
        # move the merged model to the target device in one shot.
        if torch.backends.mps.is_available():
            target_device = "mps"
            # bfloat16 support on MPS is incomplete; float16 is safer
            dtype = torch.float16
        elif torch.cuda.is_available():
            target_device = "cuda"
            dtype = torch.bfloat16
        else:
            target_device = "cpu"
            dtype = torch.bfloat16

        print(f"Loading base model {base_model!r} + adapter {adapter_repo!r}…")
        print(f"  dtype={dtype}, target_device={target_device}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            low_cpu_mem_usage=True,  # stream weights from disk, reduces peak RAM
        )
        model = PeftModel.from_pretrained(model, adapter_repo)
        model.eval()
        print(f"  Moving model to {target_device}…")
        model = model.to(target_device)
        print("  Model loaded.")
        return model, tokenizer

    def _make_stopping_criteria(self, stop: Union[str, list, None]):
        """Build a HuggingFace StoppingCriteriaList from Tinker stop sequences."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        if not stop:
            return None

        stop_list = [stop] if isinstance(stop, str) else list(stop)

        # Tokenize string stop sequences; integer sequences are used as-is
        token_stop_seqs: list[list[int]] = []
        for s in stop_list:
            if isinstance(s, str):
                ids = self._tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    token_stop_seqs.append(ids)
            elif isinstance(s, int):
                token_stop_seqs.append([s])

        if not token_stop_seqs:
            return None

        class StopOnSequence(StoppingCriteria):
            def __init__(self, seqs: list[list[int]]):
                self.seqs = [torch.tensor(s) for s in seqs]

            def __call__(self, input_ids: "torch.Tensor", scores, **kwargs) -> bool:
                for seq in self.seqs:
                    n = len(seq)
                    if input_ids.shape[1] >= n:
                        if (input_ids[0, -n:] == seq.to(input_ids.device)).all():
                            return True
                return False

        return StoppingCriteriaList([StopOnSequence(token_stop_seqs)])

    async def sample_async(
        self,
        prompt: tinker_types.ModelInput,
        num_samples: int,
        sampling_params: tinker_types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> tinker_types.SampleResponse:
        import torch

        input_ids = torch.tensor([prompt.to_ints()], dtype=torch.long).to(self._model.device)
        stopping_criteria = self._make_stopping_criteria(sampling_params.stop)

        generate_kwargs: dict = dict(
            max_new_tokens=sampling_params.max_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            do_sample=True,
            num_return_sequences=num_samples,
        )
        if sampling_params.top_k > 0:
            generate_kwargs["top_k"] = sampling_params.top_k
        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)
        if stopping_criteria:
            generate_kwargs["stopping_criteria"] = stopping_criteria

        with torch.inference_mode():
            outputs = self._model.generate(input_ids, **generate_kwargs)

        prompt_len = input_ids.shape[1]
        sequences = []
        for i in range(num_samples):
            generated_ids = outputs[i][prompt_len:].tolist()
            # Determine stop reason: "stop" if a stop sequence caused it, else "length"
            stop_reason: tinker_types.StopReason = "length"
            if stopping_criteria and sampling_params.stop:
                stop_list = [sampling_params.stop] if isinstance(sampling_params.stop, str) else list(sampling_params.stop)
                for s in stop_list:
                    if isinstance(s, str):
                        text = self._tokenizer.decode(generated_ids)
                        if s in text:
                            stop_reason = "stop"
                            break
            sequences.append(tinker_types.SampledSequence(
                stop_reason=stop_reason,
                tokens=generated_ids,
            ))

        return tinker_types.SampleResponse(sequences=sequences)


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def create_sampling_client_with_fallback(
    tinker_path: str | None,
    base_model: str,
    hf_user: str = "",
    adapter_repo: str | None = None,
    force_hf: bool = False,
    local_checkpoints_dir: Path | str | None = None,
) -> tinker.SamplingClient | HFSamplingClient:
    """
    Return a sampling client for the given checkpoint.

    Resolution order
    ----------------
    1. Local adapter directory  (fast, no network; checked under local_checkpoints_dir)
    2. Live Tinker checkpoint   (if still available)
    3. HuggingFace adapter repo (last resort)

    Args:
        tinker_path:            sampler_path from checkpoints.jsonl (tinker://...), or None
                                to use the base model directly.
        base_model:             HuggingFace model ID of the base model.
        hf_user:                HuggingFace username/org (needed for step 3).
        adapter_repo:           Override the HF adapter repo ID (auto-derived if omitted).
        force_hf:               Skip steps 1 & 2 and go straight to HuggingFace.
        local_checkpoints_dir:  Directory containing downloaded adapter weights.
                                Defaults to ~/TinkerCheckpoints if it exists.
    """
    service_client = tinker.ServiceClient()

    if tinker_path is None:
        print(f"No checkpoint — using base model: {base_model}")
        return service_client.create_sampling_client(base_model=base_model)

    if not force_hf:
        # --- Step 1: local ---
        local_dir = Path(local_checkpoints_dir) if local_checkpoints_dir else DEFAULT_LOCAL_CHECKPOINTS_DIR
        if local_dir.is_dir():
            local_path = _find_local_checkpoint(tinker_path, local_dir)
            if local_path is not None:
                print(f"Using local adapter: {local_path}")
                return HFSamplingClient(base_model=base_model, adapter_repo=str(local_path))

        # --- Step 2: Tinker ---
        if _tinker_checkpoint_alive(tinker_path):
            print(f"Tinker checkpoint available: {tinker_path}")
            return service_client.create_sampling_client(model_path=tinker_path)
        print(f"Tinker checkpoint unavailable: {tinker_path}")
    else:
        print(f"Forced HF mode — skipping local and Tinker for: {tinker_path}")

    # --- Step 3: HuggingFace ---
    repo = adapter_repo or (hf_user and _find_hf_repo_for_checkpoint(tinker_path, hf_user))
    if not repo:
        raise RuntimeError(
            f"Could not load checkpoint {tinker_path!r}.\n"
            f"  • No local adapter found under {local_checkpoints_dir or DEFAULT_LOCAL_CHECKPOINTS_DIR}\n"
            f"  • Tinker checkpoint has expired\n"
            f"  • No HuggingFace repo found (hf_user={hf_user!r})\n"
            f"Possible fixes: download weights locally, push to HF with push_checkpoints_to_hf.py, "
            f"or pass adapter_repo= explicitly."
        )
    print(f"Falling back to HuggingFace adapter: {repo}")
    return HFSamplingClient(base_model=base_model, adapter_repo=repo)
