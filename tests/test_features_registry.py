"""The builder registry: discovery skips absent siblings, and the contract bites."""

from __future__ import annotations

import polars as pl
import pytest

from nfl.data.market_blocklist import MarketColumnLeak
from nfl.features.registry import (
    BuilderContractError,
    BuilderSpec,
    Candidate,
    FeatureBuilder,
    discover_builders,
    make_builder,
)


def _play_builder(pbp: pl.DataFrame, weather: pl.DataFrame | None = None) -> pl.DataFrame:
    return pbp.select(pl.col("game_id"), pl.col("play_id"), pl.lit(1.0).alias("fake__constant"))


def _team_builder(schedules: pl.DataFrame) -> pl.DataFrame:
    home = schedules.select(pl.col("game_id"), pl.col("home_team").alias("team"))
    away = schedules.select(pl.col("game_id"), pl.col("away_team").alias("team"))
    return pl.concat([home, away]).with_columns(pl.lit(0.5).alias("tm__rating"))


def test_discovery_finds_baseline_and_skips_missing_siblings() -> None:
    found = discover_builders()
    assert "baseline_inputs" in found.names
    # Whatever has landed, nothing raises and every gap carries a reason.
    for gap in found.missing:
        assert gap.reason
        assert gap.name not in found.names


def test_missing_module_is_recorded_not_raised() -> None:
    candidates = (Candidate("ghost", "nfl.features.does_not_exist", "build_ghost", "play", "gh__"),)
    found = discover_builders(candidates=candidates)
    assert found.names == []
    assert [m.name for m in found.missing] == ["ghost"]
    assert "import failed" in found.missing[0].reason


def test_module_without_the_build_function_is_recorded() -> None:
    candidates = (Candidate("nope", "nfl.features.baseline_inputs", "build_nope", "play", "bi__"),)
    found = discover_builders(candidates=candidates)
    assert not found
    assert "has no build_nope()" in found.missing[0].reason


def test_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        discover_builders(["not_a_builder"])


def test_inputs_are_read_off_the_signature() -> None:
    spec = make_builder("fake", _play_builder, level="play", prefix="fake__")
    assert spec.required_inputs == ("pbp",)
    assert spec.optional_inputs == ("weather",)
    assert spec.missing_inputs({"schedules": None}) == ("pbp",)
    assert spec.can_run({"pbp": None})


def test_build_passes_only_the_inputs_the_builder_accepts(pbp: pl.DataFrame) -> None:
    spec = make_builder("fake", _play_builder, level="play", prefix="fake__")
    out = spec.build(pbp=pbp, schedules=pbp, something_else=pbp)
    assert out.height == pbp.height
    assert spec.feature_columns(out) == ["fake__constant"]


def test_build_without_a_required_input_raises(pbp: pl.DataFrame) -> None:
    spec = make_builder("fake", _play_builder, level="play", prefix="fake__")
    with pytest.raises(BuilderContractError, match="needs input"):
        spec.build(schedules=pbp)


def test_team_level_keys(schedules: pl.DataFrame) -> None:
    spec = make_builder("team", _team_builder, level="team", prefix="tm__")
    assert spec.key_columns == ("game_id", "team")
    out = spec.build(schedules=schedules)
    assert out.height == 2 * schedules.height


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (pl.DataFrame({"game_id": ["g"], "x__a": [1]}), "must return key column"),
        (
            pl.DataFrame({"game_id": ["g", "g"], "play_id": [1, 1], "x__a": [1, 2]}),
            "duplicate rows",
        ),
        (
            pl.DataFrame({"game_id": ["g"], "play_id": [1], "raw_yards": [3]}),
            "must prefix its features",
        ),
    ],
)
def test_contract_violations_raise(frame: pl.DataFrame, match: str) -> None:
    spec = make_builder("x", lambda: frame, level="play", prefix="x__")
    with pytest.raises(BuilderContractError, match=match):
        spec.validate(frame)


def test_market_column_in_builder_output_raises() -> None:
    frame = pl.DataFrame({"game_id": ["g"], "play_id": [1], "x__wp": [0.5], "vegas_wp": [0.5]})
    spec = make_builder("x", lambda: frame, level="play", prefix="x__")
    with pytest.raises(MarketColumnLeak):
        spec.validate(frame)


def test_passthrough_columns_are_not_contract_violations() -> None:
    frame = pl.DataFrame({"game_id": ["g"], "play_id": [1], "season": [2024], "x__a": [1.0]})
    spec = make_builder("x", lambda: frame, level="play", prefix="x__")
    assert spec.validate(frame) is frame
    assert spec.feature_columns(frame) == ["x__a"]


def test_builder_spec_satisfies_the_protocol() -> None:
    spec = make_builder("fake", _play_builder, level="play", prefix="fake__")
    assert isinstance(spec, BuilderSpec)
    assert isinstance(spec, FeatureBuilder)


def test_extra_builders_are_appended() -> None:
    spec = make_builder("fake", _play_builder, level="play", prefix="fake__")
    found = discover_builders(candidates=(), extra=(spec,))
    assert found.names == ["fake"]
    assert found.get("fake") is spec
    assert found.by_level("play") == [spec]
