import importlib.util
import sys
from pathlib import Path

import numpy as np


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "compare_sequence_cohort.py"
_SPEC = importlib.util.spec_from_file_location("compare_sequence_cohort", _MODULE_PATH)
assert _SPEC and _SPEC.loader
compare_sequence_cohort = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compare_sequence_cohort
_SPEC.loader.exec_module(compare_sequence_cohort)


def test_metrics_are_finite_and_include_expected_keys():
    result = compare_sequence_cohort._metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert set(result) == {"mae", "rmse", "r2"}
    assert result["mae"] == 0.5
