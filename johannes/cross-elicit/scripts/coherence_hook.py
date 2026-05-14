"""Helper: after a run_eval.py output dir lands in eval_results/finetuning/,
run judge_coherence_src.py on it so a coherence sidecar is produced before
anything downstream consumes the dir.

Best-effort: failures are logged and swallowed so a transient OpenRouter
outage doesn't bring down the eval orchestrator. The coherence judge can
always be re-run later — it resumes from the existing coherence_rows.jsonl.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JUDGE_PATH = os.path.join(SCRIPT_DIR, "judge_coherence_src.py")


def judge_coherence_for(
    eval_dir: str,
    judge_model: str | None = None,
    concurrency: int | None = None,
    prefix: str = "",
) -> int:
    """Run judge_coherence_src.py on a single eval_results/<run-dir> path.

    Returns the subprocess exit code. Never raises — on exception, prints the
    error and returns -1.
    """
    if not os.path.isdir(eval_dir):
        print(f"{prefix}coherence: skipping; {eval_dir!r} is not a directory")
        return -1
    cmd = [sys.executable, JUDGE_PATH, eval_dir, "--no-base"]
    if judge_model:
        cmd += ["--judge-model", judge_model]
    if concurrency is not None:
        cmd += ["--concurrency", str(concurrency)]
    try:
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"{prefix}coherence: judge_coherence_src.py exited "
                  f"{proc.returncode} on {os.path.basename(eval_dir)}")
        return proc.returncode
    except Exception as e:
        print(f"{prefix}coherence: judge_coherence_src.py raised {e!r} on "
              f"{os.path.basename(eval_dir)}")
        return -1
