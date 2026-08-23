"""Assembly: the join, the label, the meta/feature split, and the market gate."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nfl.data.market_blocklist import MarketColumnLeak, market_columns_in
from nfl.features.assemble import (
    LABEL_COLUMN,
    META_COLUMNS,
    AssemblyReport,
    assemble_dataset,
    feature_columns,
    game_context,
    home_win_label,
    read_dataset,
    write_dataset,
)
from nfl.features.registry import make_builder


@pytest.fixture
def dataset(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    return assemble_dataset([2024], pbp=pbp, schedules=schedules)


def test_one_row_per_play_with_meta_and_label(dataset: pl.DataFrame, pbp: pl.DataFrame) -> None:
    assert dataset.height == pbp.height
    assert not dataset.select(["meta__game_id", "meta__play_id"]).is_duplicated().any()
    for col in META_COLUMNS:
        assert col in dataset.columns
    assert LABEL_COLUMN in dataset.columns


def test_feature_columns_exclude_meta_and_label(dataset: pl.DataFrame) -> None:
    feats = feature_columns(dataset)
    assert feats
    assert not [c for c in feats if c.startswith(("meta__", "label__"))]
    assert set(feats) | set(
        c for c in dataset.columns if c.startswith(("meta__", "label__"))
    ) == set(dataset.columns)


def test_kickoff_and_clock_meta_are_populated(dataset: pl.DataFrame) -> None:
    assert dataset.get_column("meta__kickoff").null_count() == 0
    assert dataset.get_column("meta__game_seconds_remaining").null_count() == 0
    assert dataset.get_column("meta__period").drop_nulls().max() >= 4


def test_no_raw_play_by_play_columns_leak_through(dataset: pl.DataFrame, pbp: pl.DataFrame) -> None:
    """Only meta, label and prefixed features survive — never a raw vendor column."""
    raw = set(pbp.columns) - {"game_id", "play_id"}
    assert not raw & set(dataset.columns)


def test_market_columns_never_reach_the_dataset(
    pbp_fixture: pl.DataFrame, schedules_fixture: pl.DataFrame
) -> None:
    assert market_columns_in(pbp_fixture)
    assert market_columns_in(schedules_fixture)
    out = assemble_dataset(pbp=pbp_fixture, schedules=schedules_fixture)
    assert market_columns_in(out) == []


def test_a_builder_smuggling_a_market_column_is_rejected(
    pbp: pl.DataFrame, schedules: pl.DataFrame
) -> None:
    def leaky(pbp: pl.DataFrame) -> pl.DataFrame:
        return pbp.select("game_id", "play_id", pl.lit(0.5).alias("vegas_wp"))

    spec = make_builder("leaky", leaky, level="play", prefix="lk__")
    with pytest.raises(MarketColumnLeak):
        assemble_dataset([2024], pbp=pbp, schedules=schedules, builders=[spec])


# --------------------------------------------------------------------------- #
# Label
# --------------------------------------------------------------------------- #


def test_label_covers_win_loss_tie_and_unplayed(schedules_fixture: pl.DataFrame) -> None:
    labels = dict(
        home_win_label(schedules_fixture).iter_rows()  # (game_id, label)
    )
    assert labels["2023_01_DEN_KC"] == 1.0  # 20-10 home win
    assert labels["2024_02_KC_MIA"] == 0.0  # 20-30 home loss
    assert labels["2023_02_KC_BUF"] == 0.5  # 27-27 tie
    assert labels["2024_04_KC_BUF"] is None  # not played yet


def test_label_is_constant_within_a_game(dataset: pl.DataFrame) -> None:
    per_game = dataset.group_by("meta__game_id").agg(pl.col(LABEL_COLUMN).n_unique())
    assert per_game.get_column(LABEL_COLUMN).max() == 1


def test_label_needs_final_scores() -> None:
    with pytest.raises(KeyError):
        home_win_label(pl.DataFrame({"game_id": ["g"]}))


# --------------------------------------------------------------------------- #
# Joins
# --------------------------------------------------------------------------- #


def test_team_level_builders_are_widened_to_home_and_away(
    pbp: pl.DataFrame, schedules: pl.DataFrame
) -> None:
    def build_team(schedules: pl.DataFrame) -> pl.DataFrame:
        home = schedules.select("game_id", pl.col("home_team").alias("team")).with_columns(
            pl.lit(1.0).alias("tm__elo")
        )
        away = schedules.select("game_id", pl.col("away_team").alias("team")).with_columns(
            pl.lit(2.0).alias("tm__elo")
        )
        return pl.concat([home, away])

    spec = make_builder("team", build_team, level="team", prefix="tm__")
    out = assemble_dataset([2024], pbp=pbp, schedules=schedules, builders=[spec])
    assert "home_tm__elo" in out.columns
    assert "away_tm__elo" in out.columns
    assert out.get_column("home_tm__elo").unique().to_list() == [1.0]
    assert out.get_column("away_tm__elo").unique().to_list() == [2.0]
    assert "tm__elo" not in out.columns


def test_game_level_builders_join_on_game_id(pbp: pl.DataFrame, schedules: pl.DataFrame) -> None:
    def build_gm(schedules: pl.DataFrame) -> pl.DataFrame:
        return schedules.select("game_id", pl.lit(7.0).alias("gm__x"))

    spec = make_builder("gm", build_gm, level="game", prefix="gm__")
    out = assemble_dataset([2024], pbp=pbp, schedules=schedules, builders=[spec])
    assert out.get_column("gm__x").unique().to_list() == [7.0]


def test_absent_siblings_are_skipped_not_fatal(pbp: pl.DataFrame, schedules: pl.DataFrame) -> None:
    report = AssemblyReport()
    out = assemble_dataset([2024], pbp=pbp, schedules=schedules, report=report)
    assert out.height > 0
    # Whatever is missing today is reported with a reason and does not raise.
    assert all(reason for _, reason in report.skipped)
    assert set(report.ran) <= set(feature_prefix_owners(out))


def feature_prefix_owners(df: pl.DataFrame) -> set[str]:
    """Builder names inferable from the prefixes present, for the assertion above."""
    prefixes = {"bi__": "baseline_inputs", "ig__": "ingame", "tm__": "team", "pl__": "player"}
    owners = set()
    for col in feature_columns(df):
        bare = col.removeprefix("home_").removeprefix("away_")
        for prefix, name in prefixes.items():
            if bare.startswith(prefix):
                owners.add(name)
    return owners


def test_builder_needing_an_unavailable_input_is_skipped(
    pbp: pl.DataFrame, schedules: pl.DataFrame
) -> None:
    def needs_tracking(tracking: pl.DataFrame) -> pl.DataFrame:  # never provided
        return tracking

    spec = make_builder("tracking", needs_tracking, level="play", prefix="tk__")
    report = AssemblyReport()
    out = assemble_dataset([2024], pbp=pbp, schedules=schedules, builders=[spec], report=report)
    assert ("tracking", "missing input(s) ['tracking']") in report.skipped
    assert out.height > 0


def test_extra_inputs_reach_the_builders_that_name_them(
    pbp: pl.DataFrame, schedules: pl.DataFrame
) -> None:
    weather = pl.DataFrame({"game_id": ["2024_01_BAL_KC"], "wind": [12.0]})

    def build_wx(pbp: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
        return pbp.select("game_id", "play_id").with_columns(
            pl.lit(float(weather.get_column("wind")[0])).alias("wx__wind")
        )

    spec = make_builder("wx", build_wx, level="play", prefix="wx__")
    out = assemble_dataset(
        [2024], pbp=pbp, schedules=schedules, inputs={"weather": weather}, builders=[spec]
    )
    assert out.get_column("wx__wind").unique().to_list() == [12.0]


def test_colliding_feature_names_raise(pbp: pl.DataFrame, schedules: pl.DataFrame) -> None:
    def dupe(pbp: pl.DataFrame) -> pl.DataFrame:
        return pbp.select("game_id", "play_id", pl.lit(1.0).alias("bi__score_diff_home"))

    spec = make_builder("dupe", dupe, level="play", prefix="bi__")
    from nfl.features.registry import discover_builders

    builders = list(discover_builders(["baseline_inputs"])) + [spec]
    with pytest.raises(ValueError, match="overwrite"):
        assemble_dataset([2024], pbp=pbp, schedules=schedules, builders=builders)


# --------------------------------------------------------------------------- #
# Context and storage
# --------------------------------------------------------------------------- #


def test_game_context_is_numeric_and_one_row_per_game(schedules: pl.DataFrame) -> None:
    ctx = game_context(schedules)
    assert ctx.height == schedules.height
    assert market_columns_in(ctx) == []
    for name, dtype in ctx.schema.items():
        if name != "game_id":
            assert dtype.is_numeric()


def test_write_and_read_roundtrip_partitions_by_season(
    pbp_fixture: pl.DataFrame, schedules_fixture: pl.DataFrame, tmp_path: Path
) -> None:
    df = assemble_dataset(pbp=pbp_fixture, schedules=schedules_fixture)
    paths = write_dataset(df, root=tmp_path)
    assert {p.parent.name for p in paths} == {"season=2023", "season=2024"}
    assert all(p.name == "part-0.parquet" for p in paths)

    back = read_dataset(root=tmp_path)
    assert back.height == df.height
    assert set(back.columns) == set(df.columns)
    assert read_dataset([2024], root=tmp_path).get_column("meta__season").unique().to_list() == [
        2024
    ]


def test_assemble_can_write_directly(
    pbp: pl.DataFrame, schedules: pl.DataFrame, tmp_path: Path
) -> None:
    assemble_dataset([2024], pbp=pbp, schedules=schedules, write=True, root=tmp_path)
    assert (tmp_path / "dataset" / "season=2024" / "part-0.parquet").exists()


def test_read_dataset_without_a_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_dataset(root=tmp_path)


def test_empty_season_selection_raises(pbp: pl.DataFrame, schedules: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="no play-by-play rows"):
        assemble_dataset([1999], pbp=pbp, schedules=schedules)
