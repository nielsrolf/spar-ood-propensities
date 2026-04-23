"""Re-export from evals/eval-sensitivity/eval_sensitivity.py.

The eval lives in a hyphenated directory which can't be imported normally.
This shim provides a clean import path:

    from evals.eval_sensitivity import EvalSensitivity
"""
import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "eval-sensitivity", "eval_sensitivity.py")
_spec = importlib.util.spec_from_file_location("_eval_sensitivity_impl", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

EvalSensitivity = _mod.EvalSensitivity
