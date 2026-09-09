from pathlib import Path

import pytest

from src.opd_run_request import validate


def test_run_request_rejects_changed_source_or_configuration(tmp_path: Path) -> None:
    path = tmp_path / "run_request.json"
    validate(path, {"source_fingerprint": "a", "rounds": 4})
    validate(path, {"source_fingerprint": "a", "rounds": 4})
    with pytest.raises(RuntimeError, match="OUTPUT_ROOT"):
        validate(path, {"source_fingerprint": "b", "rounds": 4})
