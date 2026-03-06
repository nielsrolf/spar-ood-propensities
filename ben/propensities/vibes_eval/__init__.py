"""
VisEval: A library for visualizing model evaluation results.

This package provides tools for visualizing and comparing model evaluation results,
with a focus on language model evaluations.
"""

import logging

from .vibes_eval import VisEval as VisEval, VisEvalResult as VisEvalResult
from .freeform import FreeformQuestion as FreeformQuestion, FreeformEval as FreeformEval
from .runner import dispatcher as dispatcher

try:
    from .runner import LocalRouterRunner as LocalRouterRunner
except ImportError:
    pass

try:
    # pyrefly: ignore [missing-import]
    from .multiple_choice import MCEvalRunner as MCEvalRunner
except ImportError:
    pass

logging.getLogger("cache_on_disk").setLevel(logging.ERROR)

__version__ = "0.1.0"
