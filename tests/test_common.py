from src.common import SCENE_NAMES


def test_scene_names_are_unique_and_complete() -> None:
    assert len(SCENE_NAMES) == 5
    assert len(set(SCENE_NAMES)) == len(SCENE_NAMES)
