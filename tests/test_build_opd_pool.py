from pathlib import Path

from src.build_opd_pool import build_pool
from src.common import read_jsonl, write_jsonl


def _rows(prefix: str, count: int) -> list[dict]:
    return [
        {"source_id": f"{prefix}-{index}", "audio_path": f"{prefix}-{index}.wav", "scene": "clean", "text": "字"}
        for index in range(count)
    ]


def test_pool_reserves_source_disjoint_calibration_split(tmp_path: Path) -> None:
    train, validation, test = (tmp_path / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl"))
    write_jsonl(_rows("train", 4), train)
    write_jsonl(_rows("validation", 10), validation)
    write_jsonl(_rows("test", 3), test)
    pool, calibration, report = tmp_path / "pool.jsonl", tmp_path / "calibration.jsonl", tmp_path / "report.json"
    result = build_pool(train, validation, test, pool, report, calibration, 0.2)
    assert result["calibration_rows"] == 2
    assert len(read_jsonl(pool)) == 12
    pool_ids = {row["source_id"] for row in read_jsonl(pool)}
    calibration_ids = {row["source_id"] for row in read_jsonl(calibration)}
    assert pool_ids.isdisjoint(calibration_ids)
