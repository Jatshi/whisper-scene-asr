from src.common import stratified_limit


def test_stratified_limit_is_balanced_and_deterministic() -> None:
    rows = [
        {"audio_path": f"/{scene}/{index}.wav", "source_id": f"{scene}-{index}", "scene": scene}
        for scene in ("clean", "noisy", "reverb", "fast_slow", "noisy_reverb")
        for index in range(20)
    ]
    selected = stratified_limit(rows, 25)
    counts = {scene: sum(row["scene"] == scene for row in selected) for scene in {row["scene"] for row in rows}}
    assert counts == {scene: 5 for scene in counts}
    assert selected == stratified_limit(rows, 25)
    assert selected != rows[:25]
