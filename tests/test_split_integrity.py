import pytest

from src.common import assert_disjoint_sources


def test_source_views_cannot_cross_splits() -> None:
    with pytest.raises(ValueError, match="Source leakage"):
        assert_disjoint_sources([{"source_id": "u1"}], [{"source_id": "u1"}])


def test_disjoint_sources_pass() -> None:
    assert_disjoint_sources([{"source_id": "u1"}], [{"source_id": "u2"}], [{"source_id": "u3"}])
