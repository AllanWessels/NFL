"""The feature-builder registry: what a builder is, and which ones exist today.

Phase 4 splits the feature work across independently developed modules
(``ingame``, ``team``, ``player``, ...). Assembly must not care which of them
have landed: a missing module is a *skipped* builder, not an import error. That
is the whole job of this module.

A builder is a module-level function following the Phase 4 contract:

* ``nfl/features/<name>.py`` exposes ``build_<name>(...) -> polars.DataFrame``
* it returns one row per ``(game_id, play_id)`` (level ``"play"``), per
  ``(game_id, team)`` (level ``"team"``) or per ``game_id`` (level ``"game"``)
* every non-key column is named ``<prefix>__<feature>``
* it calls :func:`nfl.data.market_blocklist.assert_no_market_columns` itself

The inputs a builder wants are read off its **signature**, not off a hand
maintained list: :func:`discover_builders` introspects the function and asks for
the frames it names (``pbp``, ``schedules``, ``weather``, ...). Parameters with
defaults are optional; parameters without are required, and a builder whose
required inputs are unavailable is skipped with a reason rather than crashing
the run. This keeps assembly working while three siblings are still landing
their signatures.

    >>> found = discover_builders()
    >>> [b.name for b in found]                       # doctest: +SKIP
    ['baseline_inputs', 'ingame', 'team', 'player']
    >>> [(m.name, m.reason) for m in found.missing]   # doctest: +SKIP
    [('player', "import failed: No module named 'nfl.features.player'")]
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, Protocol, runtime_checkable

import polars as pl

from nfl.data.market_blocklist import assert_no_market_columns

__all__ = [
    "CANDIDATES",
    "KEY_COLUMNS",
    "PASSTHROUGH_COLUMNS",
    "BuilderContractError",
    "BuilderSpec",
    "Discovery",
    "FeatureBuilder",
    "Level",
    "MissingBuilder",
    "discover_builders",
    "make_builder",
]

#: The grain a builder returns. ``"play"`` frames join on ``(game_id, play_id)``;
#: ``"team"`` frames are pivoted to ``home_``/``away_`` columns at assembly time.
Level = Literal["play", "team", "game"]

#: Key columns each level must return.
KEY_COLUMNS: Mapping[Level, tuple[str, ...]] = {
    "play": ("game_id", "play_id"),
    "team": ("game_id", "team"),
    "game": ("game_id",),
}

#: Columns a builder may return that are neither keys nor features. They are
#: dropped at assembly time instead of being treated as unprefixed features, so
#: a builder that helpfully carries ``season`` along is not a contract error.
PASSTHROUGH_COLUMNS: frozenset[str] = frozenset(
    {
        "game_id",
        "play_id",
        "team",
        "season",
        "season_type",
        "game_type",
        "week",
        "kickoff",
        "posteam",
        "defteam",
        "home_team",
        "away_team",
        "qtr",
        "period",
        "game_seconds_remaining",
        "source",
        "ingested_at",
        "schema_version",
    }
)


class BuilderContractError(AssertionError):
    """A builder returned a frame that violates the Phase 4 contract."""


@runtime_checkable
class FeatureBuilder(Protocol):
    """What assembly needs from a feature builder.

    :class:`BuilderSpec` is the implementation; the protocol exists so a caller
    can supply its own object (a class with a ``build`` method, say) without
    importing anything from here.
    """

    name: str
    level: Level
    prefix: str
    required_inputs: tuple[str, ...]

    def build(self, **inputs: pl.DataFrame) -> pl.DataFrame:
        """Return the builder's frame, given the named input frames."""


@dataclass(frozen=True, slots=True)
class BuilderSpec:
    """A discovered builder: where it came from and what it needs."""

    name: str
    level: Level
    prefix: str
    module: str
    fn: Callable[..., pl.DataFrame]
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    accepts_any_input: bool = False

    @property
    def key_columns(self) -> tuple[str, ...]:
        return KEY_COLUMNS[self.level]

    @property
    def accepted_inputs(self) -> tuple[str, ...]:
        return self.required_inputs + self.optional_inputs

    def missing_inputs(self, available: Sequence[str] | Mapping[str, object]) -> tuple[str, ...]:
        """Required inputs that ``available`` does not provide."""
        have = set(available)
        return tuple(name for name in self.required_inputs if name not in have)

    def can_run(self, available: Sequence[str] | Mapping[str, object]) -> bool:
        return not self.missing_inputs(available)

    def build(self, **inputs: pl.DataFrame) -> pl.DataFrame:
        """Call the builder with the subset of ``inputs`` it accepts.

        Raises :class:`BuilderContractError` if a required input is missing or
        the returned frame breaks the key/prefix contract, and
        :class:`~nfl.data.market_blocklist.MarketColumnLeak` if it carries a
        market column (the builder is required to check this itself; assembly
        checks again because the cost of missing it is the whole project).
        """
        missing = self.missing_inputs(inputs)
        if missing:
            raise BuilderContractError(
                f"builder {self.name!r} needs input(s) {list(missing)}; got {sorted(inputs)}"
            )
        kwargs = (
            dict(inputs)
            if self.accepts_any_input
            else {k: v for k, v in inputs.items() if k in self.accepted_inputs}
        )
        out = self.fn(**kwargs)
        self.validate(out)
        return out

    def validate(self, out: pl.DataFrame) -> pl.DataFrame:
        """Check the market blocklist, keys, key uniqueness and column prefixes.

        The market gate goes first: an unprefixed ``vegas_wp`` is a market leak
        before it is a naming problem, and that is the error worth reporting.
        """
        if not isinstance(out, pl.DataFrame):
            raise BuilderContractError(
                f"builder {self.name!r} returned {type(out).__name__}, expected polars.DataFrame"
            )
        assert_no_market_columns(out, context=f"builder {self.name!r}")
        missing_keys = [k for k in self.key_columns if k not in out.columns]
        if missing_keys:
            raise BuilderContractError(
                f"builder {self.name!r} ({self.level}-level) must return key column(s) "
                f"{list(self.key_columns)}; missing {missing_keys}"
            )
        keys = list(self.key_columns)
        if out.select(keys).is_duplicated().any():
            raise BuilderContractError(
                f"builder {self.name!r} returned duplicate rows for key {keys}"
            )
        stray = [
            c for c in out.columns if c not in PASSTHROUGH_COLUMNS and not c.startswith(self.prefix)
        ]
        if stray:
            raise BuilderContractError(
                f"builder {self.name!r} must prefix its features with {self.prefix!r}; "
                f"unprefixed column(s): {sorted(stray)}"
            )
        return out

    def feature_columns(self, out: pl.DataFrame) -> list[str]:
        """The columns of ``out`` that are actual features (prefix-matched)."""
        return [c for c in out.columns if c.startswith(self.prefix)]


@dataclass(frozen=True, slots=True)
class MissingBuilder:
    """A candidate builder that could not be loaded, and why."""

    name: str
    module: str
    reason: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """A builder module assembly will look for."""

    name: str
    module: str
    attr: str
    level: Level
    prefix: str


#: The Phase 4 builders assembly knows how to look for. Order is join order.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "baseline_inputs",
        "nfl.features.baseline_inputs",
        "build_baseline_inputs",
        "play",
        "bi__",
    ),
    Candidate("ingame", "nfl.features.ingame", "build_ingame", "play", "ig__"),
    Candidate("team", "nfl.features.team", "build_team", "team", "tm__"),
    Candidate("player", "nfl.features.player", "build_player", "team", "pl__"),
)


def _signature_inputs(
    fn: Callable[..., pl.DataFrame],
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """``(required, optional, accepts_any)`` input names read off ``fn``'s signature."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables: assume nothing
        return (), (), True
    required: list[str] = []
    optional: list[str] = []
    accepts_any = False
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_any = True
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
        else:
            optional.append(param.name)
    return tuple(required), tuple(optional), accepts_any


def make_builder(
    name: str,
    fn: Callable[..., pl.DataFrame],
    *,
    level: Level,
    prefix: str | None = None,
    module: str | None = None,
) -> BuilderSpec:
    """Wrap a plain ``build_*`` function into a :class:`BuilderSpec`.

    Used by :func:`discover_builders` and by tests that want to register a
    stand-in for a sibling module that has not landed yet.
    """
    required, optional, accepts_any = _signature_inputs(fn)
    return BuilderSpec(
        name=name,
        level=level,
        prefix=prefix if prefix is not None else f"{name}__",
        module=module or str(getattr(fn, "__module__", None) or "<callable>"),
        fn=fn,
        required_inputs=required,
        optional_inputs=optional,
        accepts_any_input=accepts_any,
    )


@dataclass(frozen=True, slots=True)
class Discovery:
    """The result of :func:`discover_builders`: what loaded, and what did not."""

    builders: tuple[BuilderSpec, ...] = ()
    missing: tuple[MissingBuilder, ...] = ()

    def __iter__(self) -> Iterator[BuilderSpec]:
        return iter(self.builders)

    def __len__(self) -> int:
        return len(self.builders)

    def __bool__(self) -> bool:
        return bool(self.builders)

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.builders]

    def by_level(self, level: Level) -> list[BuilderSpec]:
        return [b for b in self.builders if b.level == level]

    def get(self, name: str) -> BuilderSpec | None:
        return next((b for b in self.builders if b.name == name), None)


def discover_builders(
    names: Sequence[str] | None = None,
    *,
    extra: Sequence[BuilderSpec] = (),
    candidates: Sequence[Candidate] = CANDIDATES,
) -> Discovery:
    """Import every known builder module that exists; skip the rest gracefully.

    ``names`` restricts discovery to a subset (unknown names raise). ``extra``
    appends already-built specs — how a test injects a fake sibling.
    """
    wanted = list(candidates)
    if names is not None:
        known = {c.name for c in candidates}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise KeyError(f"unknown builder(s) {unknown}; known: {sorted(known)}")
        wanted = [c for c in candidates if c.name in set(names)]

    builders: list[BuilderSpec] = []
    missing: list[MissingBuilder] = []
    for cand in wanted:
        try:
            module = import_module(cand.module)
        except ImportError as exc:  # sibling has not landed yet
            missing.append(MissingBuilder(cand.name, cand.module, f"import failed: {exc}"))
            continue
        fn = getattr(module, cand.attr, None)
        if fn is None:
            missing.append(MissingBuilder(cand.name, cand.module, f"module has no {cand.attr}()"))
            continue
        if not callable(fn):
            missing.append(MissingBuilder(cand.name, cand.module, f"{cand.attr} is not callable"))
            continue
        builders.append(
            make_builder(cand.name, fn, level=cand.level, prefix=cand.prefix, module=cand.module)
        )
    builders.extend(extra)
    return Discovery(tuple(builders), tuple(missing))
