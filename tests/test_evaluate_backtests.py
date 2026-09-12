import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_backtests.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_backtests", _MODULE_PATH)
assert _SPEC and _SPEC.loader
evaluate_backtests = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_backtests)


def test_folds_are_expanding_and_three_month_windows():
    folds = evaluate_backtests._folds()
    assert len(folds) == 5
    assert folds[0]["train_end_utc"] < folds[1]["train_end_utc"]
    assert folds[0]["test_end_utc"] > folds[0]["train_end_utc"]
