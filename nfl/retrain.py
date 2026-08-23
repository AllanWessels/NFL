"""The whole weekly loop as one command: new football in, promotion decision out.

``nfl retrain`` chains the steps that were previously nine invocations typed in
the right order, which is nine chances to run them in the wrong one. The order
is not cosmetic — each step consumes what the one before it wrote:

.. code-block:: text

    ingest nflverse  ->  raw pbp + schedules for the current season
    ingest weather   ->  needs those schedules to know where and when to fetch
    features build   ->  needs both; rebuilt over every season, not just the new
                         one, because a rolling split moves the boundaries and a
                         season that was val last week is train this week
    train naive      ->  the floor, first: its probability band keys the
                         not-collapsed slice every later model is scored on
    train xgb        ->  sweep-best params, read back out of the latest manifest
    train seq        ->  likewise
    train ensemble   ->  loads the three from the registry, fits meta+calibrator
    promote          ->  ADR 0003 §5 gate; no --force, ever, from here
    ledger resolve   ->  fill in outcomes for games that finished
    evaluate         ->  score the ledger, surface the drift flag

**Two kinds of not-succeeding, deliberately different.** A step that *raises* —
no network, no dataset, a trainer that blew up — stops the chain then and there:
everything downstream would be operating on stale or absent inputs. A gate that
*refuses* is not an error, it is the gate doing its job, so the chain carries on
into ledger resolve and evaluate (which describe the model already in
production, which is still the one being served) and the command exits non-zero
at the end with the refusal named. Losing the drift reading because a candidate
failed to beat the floor would be exactly backwards.

Every step is timed, and the whole run is written to
``data/reports/retrain-<timestamp>.md`` whether it succeeded or not — a failed
run is the one whose log you actually want.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nfl import config
from nfl.features.splits import SeasonSplit, resolve_split

log = logging.getLogger(__name__)

__all__ = [
    "RETRAIN_STEPS",
    "RetrainResult",
    "StepResult",
    "current_season",
    "recorded_seq_config",
    "recorded_xgb_config",
    "run_retrain",
    "summary_table",
    "to_markdown",
    "write_report",
]

#: The chain, in the only order that works. Exposed as data so a test can assert
#: the ordering without running a single fit.
RETRAIN_STEPS: tuple[str, ...] = (
    "ingest nflverse",
    "ingest weather",
    "features build",
    "train naive",
    "train xgb",
    "train seq",
    "train ensemble",
    "promote",
    "ledger resolve",
    "evaluate",
)

#: The two steps ``--skip-ingest`` drops.
INGEST_STEPS: tuple[str, ...] = ("ingest nflverse", "ingest weather")

#: The model the chain trains, gates and promotes.
CANDIDATE_MODEL = "ensemble_v1"

#: Feature set for the servable models (ADR 0003): in-play state minus the
#: columns a live snapshot cannot fill. Only used when a manifest does not
#: already say which set its params were swept on.
DEFAULT_FEATURE_SET = "servable"

StepStatus = Literal["ok", "skipped", "failed", "refused", "planned"]


@dataclass
class StepResult:
    """What one link of the chain did, and how long it took."""

    name: str
    status: StepStatus = "planned"
    seconds: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped", "planned"}


@dataclass
class RetrainResult:
    """The whole run: every step, the gate verdict, the pointer, the drift flag."""

    started_at: datetime
    season: int
    split: SeasonSplit
    dry_run: bool = False
    steps: list[StepResult] = field(default_factory=list)
    #: :class:`~nfl.models.train_ensemble.GateResult`, when the gate ran.
    gate: Any = None
    promoted: bool = False
    #: ``data/models/PRODUCTION.json`` as it stands *after* the run.
    pointer: dict[str, Any] | None = None
    #: :class:`~nfl.logging.evaluate.RetrainSignal` for the production model.
    drift: Any = None
    candidate: str = ""
    candidate_version: str = ""
    report_path: Path | None = None
    seconds: float = 0.0

    @property
    def failed_step(self) -> StepResult | None:
        """The step that raised, if one did."""
        return next((s for s in self.steps if s.status == "failed"), None)

    @property
    def succeeded(self) -> bool:
        """No step raised and nothing refused. A dry run counts as a success."""
        return all(s.ok for s in self.steps)

    def step(self, name: str) -> StepResult | None:
        return next((s for s in self.steps if s.name == name), None)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def current_season(today: datetime | None = None) -> int:
    """The NFL season in progress (or next up) on ``today``.

    A season is named for the calendar year it kicks off in and runs into
    February, so January and February belong to the previous year's season;
    March onwards is already the next one, which is when nflverse starts
    publishing its schedule.
    """
    now = today or datetime.now(UTC)
    return now.year if now.month >= 3 else now.year - 1


def _latest_manifest(name: str, root: Path | None = None) -> dict[str, Any] | None:
    """The newest manifest for ``name``, flattened, or ``None`` if there is none."""
    from nfl.models import registry

    try:
        version = registry.latest(name, root)
    except FileNotFoundError:
        return None
    manifest = registry.read_manifest(registry.model_dir(name, version, root))
    payload = dict(manifest.extra)
    payload["version"] = manifest.version
    payload["metrics"] = dict(manifest.metrics)
    return payload


def recorded_xgb_config(root: Path | None = None) -> dict[str, Any]:
    """Booster params and training switches from the latest XGBoost manifest.

    The sweeps under ``data/models/sweeps/`` picked these; the manifest of the
    artifact they produced is where they were written down. Reading them back
    is what makes a weekly retrain a *refit* of a known configuration rather
    than a fresh roll of whatever the module defaults happen to be.
    """
    manifest = _latest_manifest("xgboost_winprob", root) or {}
    params = manifest.get("params")
    return {
        "params": dict(params) if isinstance(params, dict) else {},
        "feature_set": str(manifest.get("feature_set") or DEFAULT_FEATURE_SET),
        "selection_metric": str(manifest.get("selection_metric") or "q4"),
        "weight_by_game": bool(manifest.get("weight_by_game", True)),
        "source_version": manifest.get("version", ""),
    }


def recorded_seq_config(root: Path | None = None) -> dict[str, Any]:
    """GRU hyperparameters from the latest ``tf_seq`` manifest, same reasoning."""
    manifest = _latest_manifest("tf_seq", root) or {}
    hyper = manifest.get("hyperparams")
    kwargs = dict(hyper) if isinstance(hyper, dict) else {}
    kwargs.setdefault("feature_set", str(manifest.get("feature_set") or DEFAULT_FEATURE_SET))
    kwargs.setdefault("selection_metric", str(manifest.get("selection_metric") or "q4"))
    return {
        "kwargs": kwargs,
        "weight_by_game": bool(manifest.get("weight_by_game", True)),
        "source_version": manifest.get("version", ""),
    }


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #


class _Runner:
    """Times each step, records it, and stops the chain when one raises."""

    def __init__(self, result: RetrainResult, echo: Callable[[str], None]) -> None:
        self.result = result
        self.echo = echo
        self.stopped = False

    def skip(self, name: str, why: str) -> None:
        self.result.steps.append(StepResult(name, "skipped", 0.0, why))
        self.echo(f"[dim]skip[/dim]  {name} — {why}")

    def plan(self, name: str, detail: str = "") -> None:
        self.result.steps.append(StepResult(name, "planned", 0.0, detail))
        self.echo(f"[dim]plan[/dim]  {name}" + (f" — {detail}" if detail else ""))

    @contextmanager
    def step(self, name: str) -> Iterator[StepResult]:
        entry = StepResult(name, "ok")
        self.result.steps.append(entry)
        self.echo(f"[bold]->[/bold] {name}")
        started = time.perf_counter()
        try:
            yield entry
        # Recorded and swallowed on purpose: the chain stops via ``stopped``, and
        # the run log — the thing you actually want after a failure — still gets
        # written by the caller.
        except Exception as exc:  # noqa: BLE001
            entry.status = "failed"
            entry.seconds = time.perf_counter() - started
            entry.detail = f"{type(exc).__name__}: {exc}"
            self.stopped = True
            log.exception("retrain step %r failed", name)
            self.echo(f"[red]FAILED[/red] {name} after {entry.seconds:.1f}s — {entry.detail}")
            return
        entry.seconds = time.perf_counter() - started
        self.echo(f"[green]ok[/green]    {name} ({entry.seconds:.1f}s) {entry.detail}".rstrip())


def run_retrain(
    *,
    season: int | None = None,
    split: SeasonSplit | str | None = None,
    skip_ingest: bool = False,
    dry_run: bool = False,
    echo: Callable[[str], None] | None = None,
    models_root: Path | None = None,
    data_root: Path | None = None,
    parity_samples: int | None = None,
    write_log: bool = True,
) -> RetrainResult:
    """Run the chain and return everything it learned. Never raises for a step.

    Args:
        season: the season to ingest; defaults to :func:`current_season`.
        split: season split for every fit; ``None`` resolves the rolling default.
        skip_ingest: drop the two ingest steps (offline reruns, and the way to
            retrain without waiting on the vendor).
        dry_run: print the plan — the resolved season, split and step order —
            and touch nothing.
        echo: sink for progress lines (rich markup); defaults to the logger.
        parity_samples: override the gate's serving-parity sample count.
        write_log: write ``data/reports/retrain-<ts>.md``.
    """
    started = datetime.now(UTC)
    say = echo or (lambda line: log.info("%s", line))
    resolved_season = season if season is not None else current_season(started)
    resolved_split = resolve_split(split)

    result = RetrainResult(
        started_at=started,
        season=resolved_season,
        split=resolved_split,
        dry_run=dry_run,
        candidate=CANDIDATE_MODEL,
    )
    say(
        f"[bold]retrain[/bold] season={resolved_season} split={resolved_split.label} "
        f"{'(dry run)' if dry_run else ''}".rstrip()
    )
    runner = _Runner(result, say)
    wall = time.perf_counter()

    if dry_run:
        for name in RETRAIN_STEPS:
            if skip_ingest and name in INGEST_STEPS:
                runner.skip(name, "--skip-ingest")
            else:
                runner.plan(name, _plan_detail(name, result, models_root))
        result.seconds = time.perf_counter() - wall
        result.pointer = _read_pointer(models_root)
        if write_log:
            result.report_path = write_report(result, root=data_root)
        return result

    _ingest(runner, result, skip_ingest=skip_ingest, models_root=models_root)
    # The ingest may have completed a regular season, which moves the rolling
    # window by one. Re-resolving here — but only when the caller did not pin a
    # split — is the difference between training on the football that just
    # arrived and training on last week's idea of what "latest" meant.
    if split is None and not runner.stopped:
        rerolled = resolve_split(None)
        if rerolled != result.split:
            say(
                f"[yellow]split moved[/yellow] {result.split.label} -> {rerolled.label} "
                "(a regular season completed in this ingest)"
            )
            result.split = rerolled
    if not runner.stopped:
        _build_features(runner, result)
    if not runner.stopped:
        _train(runner, result, models_root=models_root)
    if not runner.stopped:
        _promote(runner, result, models_root=models_root, parity_samples=parity_samples)
    # Ledger resolve and evaluate describe what is *already* in production, so a
    # refused gate must not cost us the drift reading. A raised step still stops.
    if not runner.stopped:
        _resolve_and_evaluate(runner, result, models_root=models_root)

    result.pointer = _read_pointer(models_root)
    result.seconds = time.perf_counter() - wall
    if write_log:
        result.report_path = write_report(result, root=data_root)
    return result


def _plan_detail(name: str, result: RetrainResult, models_root: Path | None) -> str:
    """What a dry run can say about a step without doing any of it."""
    if name == "ingest nflverse":
        return f"seasons {result.season} (idempotent)"
    if name == "ingest weather":
        return f"season {result.season}"
    if name == "features build":
        return "all seasons on disk"
    if name == "train xgb":
        cfg = recorded_xgb_config(models_root)
        return f"feature_set={cfg['feature_set']} params from {cfg['source_version'] or '(none)'}"
    if name == "train seq":
        cfg = recorded_seq_config(models_root)
        return (
            f"feature_set={cfg['kwargs'].get('feature_set')} "
            f"params from {cfg['source_version'] or '(none)'}"
        )
    if name == "train ensemble":
        return "logit_mean + phase-aware isotonic"
    if name == "promote":
        return f"{CANDIDATE_MODEL} through the ADR 0003 §5 gate (no --force)"
    return ""


def _ingest(
    runner: _Runner, result: RetrainResult, *, skip_ingest: bool, models_root: Path | None
) -> None:
    if skip_ingest:
        runner.skip("ingest nflverse", "--skip-ingest")
        runner.skip("ingest weather", "--skip-ingest")
        return

    with runner.step("ingest nflverse") as entry:
        from nfl.data.nflverse import IngestResult, ingest_nflverse
        from nfl.features.splits import clear_split_cache

        results: list[IngestResult] = []
        ingest_nflverse([result.season], force=False, results=results)
        written = sum(r.rows for r in results if r.written)
        entry.detail = f"{written:,} row(s) written across {len(results)} table(s)"
        # The rolling split is dated from these schedules; the cached scan is
        # now a week out of date.
        clear_split_cache()
    if runner.stopped:
        return

    with runner.step("ingest weather") as entry:
        from nfl.data import weather as weather_mod

        paths = weather_mod.ingest_weather(
            str(result.season),
            games_parquet=weather_mod.DEFAULT_GAMES_PARQUET,
            mode="auto",
            force=False,
        )
        entry.detail = f"{len(paths)} season parquet(s)"


def _build_features(runner: _Runner, result: RetrainResult) -> None:
    with runner.step("features build") as entry:
        from nfl.features.assemble import AssemblyReport, assemble_dataset, feature_columns

        report = AssemblyReport()
        # Every season, not only the new one: the split boundaries moved, so
        # rows that were val last week are train this week and must have been
        # built by the same builders at the same version.
        frame = assemble_dataset(None, write=True, report=report)
        games = frame.get_column("meta__game_id").n_unique()
        entry.detail = (
            f"{frame.height:,} rows / {games:,} games / {len(feature_columns(frame)):,} features"
        )
        if report.skipped:
            entry.detail += f" ({len(report.skipped)} builder(s) skipped)"


def _train(runner: _Runner, result: RetrainResult, *, models_root: Path | None) -> None:
    split = result.split

    with runner.step("train naive") as entry:
        from nfl.models.naive import train_naive

        _, res = train_naive(split=split, models_root=models_root)
        entry.detail = _metric_line(res)
    if runner.stopped:
        return

    with runner.step("train xgb") as entry:
        from nfl.models.gbdt import train_xgboost

        cfg = recorded_xgb_config(models_root)
        _, res = train_xgboost(
            split=split,
            models_root=models_root,
            params=cfg["params"],
            feature_set=cfg["feature_set"],
            selection_metric=cfg["selection_metric"],
            weight_by_game=cfg["weight_by_game"],
        )
        entry.detail = f"{_metric_line(res)} [params<-{cfg['source_version'] or 'defaults'}]"
    if runner.stopped:
        return

    with runner.step("train seq") as entry:
        from nfl.models.tf_seq import train_tf_seq

        cfg = recorded_seq_config(models_root)
        _, res = train_tf_seq(
            split=split,
            models_root=models_root,
            weight_by_game=cfg["weight_by_game"],
            **cfg["kwargs"],
        )
        entry.detail = f"{_metric_line(res)} [params<-{cfg['source_version'] or 'defaults'}]"
    if runner.stopped:
        return

    with runner.step("train ensemble") as entry:
        from nfl.models.train_ensemble import train_ensemble

        ensemble, res = train_ensemble(
            meta="logit_mean",
            calibrator="phase_aware",
            calibrator_base="isotonic",
            split=split,
            models_root=models_root,
        )
        result.candidate = ensemble.name
        result.candidate_version = str(res.version or "")
        entry.detail = f"{ensemble.name}/{res.version} {_metric_line(res)}"


def _metric_line(result: Any) -> str:
    """``log_loss=… brier=…`` from a ``FitResult``, whichever way it spells them.

    Trainers that report both sides of the fit prefix their keys (``val_brier``);
    trainers whose only metrics are already the validation ones do not. Look for
    the prefixed name first so the number in the step line is always the
    held-out one.
    """
    metrics = getattr(result, "metrics", None) or {}
    parts = []
    for metric in ("log_loss", "brier"):
        for key in (f"val_{metric}", metric):
            if key in metrics:
                parts.append(f"{metric}={float(metrics[key]):.4f}")
                break
    return " ".join(parts)


def _promote(
    runner: _Runner,
    result: RetrainResult,
    *,
    models_root: Path | None,
    parity_samples: int | None,
) -> None:
    with runner.step("promote") as entry:
        from nfl.models import registry
        from nfl.models.train_ensemble import (
            SERVING_PARITY_SAMPLES,
            promotion_gate,
        )

        gate = promotion_gate(
            result.candidate,
            result.candidate_version or None,
            root=models_root,
            parity_samples=(SERVING_PARITY_SAMPLES if parity_samples is None else parity_samples),
        )
        result.gate = gate
        result.candidate_version = gate.candidate_version
        failed = [f"{c.slice}/{c.metric}" for c in gate.checks if not c.passed]
        if gate.passed:
            registry.promote(gate.candidate, gate.candidate_version, root=models_root)
            result.promoted = True
            entry.detail = f"PASS on all {len(gate.checks)} checks -> PRODUCTION"
        else:
            # Not an error: the gate refusing is the gate working. The chain
            # continues so the drift reading on the *current* production model
            # still gets taken.
            entry.status = "refused"
            entry.detail = f"gate FAILED {len(failed)}/{len(gate.checks)}: {', '.join(failed)}"


def _resolve_and_evaluate(
    runner: _Runner, result: RetrainResult, *, models_root: Path | None
) -> None:
    with runner.step("ledger resolve") as entry:
        import polars as pl

        from nfl.data.market import closing_lines
        from nfl.features.assemble import load_raw_table
        from nfl.logging import ledger as ledger_mod

        try:
            schedules = load_raw_table("schedules")
        except FileNotFoundError:
            path = closing_lines.find_schedule_file()
            if path is None:
                raise FileNotFoundError(
                    "no nflverse schedule on disk; run `nfl ingest nflverse --tables schedules`"
                ) from None
            schedules = pl.read_csv(path) if path.suffix == ".csv" else pl.read_parquet(path)
        report = ledger_mod.resolve_ledger(schedules)
        entry.detail = f"{report.newly_resolved:,} newly resolved, {report.still_open:,} still open"
    if runner.stopped:
        return

    with runner.step("evaluate") as entry:
        from nfl.logging import evaluate as evaluate_mod

        report = evaluate_mod.evaluate_ledger(models_root=models_root)
        result.drift = _production_drift(report, result, models_root)
        entry.detail = f"{report.n_rows:,} ledger row(s), {len(report.models)} model version(s)"


def _production_drift(report: Any, result: RetrainResult, models_root: Path | None) -> Any:
    """The drift signal for whatever PRODUCTION points at, else the worst one."""
    pointer = _read_pointer(models_root)
    signals = [m.retrain for m in report.models if m.retrain is not None]
    if pointer:
        for model in report.models:
            if (
                model.model_name == pointer.get("name")
                and model.model_version == pointer.get("version")
                and model.retrain is not None
            ):
                return model.retrain
    triggered = [s for s in signals if s.triggered]
    if triggered:
        return max(triggered, key=lambda s: s.drift)
    return signals[0] if signals else None


def _read_pointer(models_root: Path | None) -> dict[str, Any] | None:
    from nfl.models import registry

    try:
        return registry.read_pointer(root=models_root)
    except (OSError, ValueError):  # pragma: no cover - unreadable pointer
        return None


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _gate_cell(result: RetrainResult) -> str:
    if result.dry_run:
        return "not run (dry run)"
    if result.gate is None:
        step = result.failed_step
        return f"not reached ({step.name} failed)" if step else "not run"
    if result.gate.passed:
        return f"PASS ({len(result.gate.checks)}/{len(result.gate.checks)} checks)"
    failed = [f"{c.slice}/{c.metric}" for c in result.gate.checks if not c.passed]
    return f"FAIL — {', '.join(failed)}"


def _pointer_cell(result: RetrainResult) -> str:
    if not result.pointer:
        return "nothing promoted"
    moved = " (moved this run)" if result.promoted else " (unchanged)"
    return f"{result.pointer.get('name')}/{result.pointer.get('version')}{moved}"


def _drift_cell(result: RetrainResult) -> str:
    signal = result.drift
    if signal is None:
        return "no signal — no resolved predictions with a training baseline"
    flag = "RETRAIN" if signal.triggered else "ok"
    return (
        f"{flag} — recent {signal.recent_log_loss:.4f} vs baseline "
        f"{signal.baseline_log_loss:.4f} (drift {signal.drift:+.4f}, "
        f"threshold {signal.threshold:.4f}, n={signal.n:,})"
    )


def summary_table(result: RetrainResult) -> list[tuple[str, str]]:
    """``(label, value)`` rows: the three answers a retrain run exists to give."""
    rows = [
        ("season", str(result.season)),
        ("split", result.split.label),
        ("candidate", f"{result.candidate}/{result.candidate_version or '—'}"),
        ("gate", _gate_cell(result)),
        ("PRODUCTION", _pointer_cell(result)),
        ("drift", _drift_cell(result)),
        ("wall clock", f"{result.seconds:.1f}s"),
    ]
    failed = result.failed_step
    if failed:
        rows.append(("stopped at", f"{failed.name} — {failed.detail}"))
    return rows


def render_summary(result: RetrainResult, console: Any) -> None:
    """Print :func:`summary_table` plus the per-step timings to a rich console."""
    from rich.table import Table

    steps = Table(title="retrain steps")
    steps.add_column("step")
    steps.add_column("status")
    steps.add_column("seconds", justify="right")
    steps.add_column("detail")
    colours = {
        "ok": "green",
        "skipped": "dim",
        "planned": "cyan",
        "refused": "yellow",
        "failed": "red",
    }
    for step in result.steps:
        colour = colours.get(step.status, "white")
        steps.add_row(
            step.name,
            f"[{colour}]{step.status}[/{colour}]",
            f"{step.seconds:.1f}",
            step.detail,
        )
    console.print(steps)

    table = Table(title="retrain summary")
    table.add_column("")
    table.add_column("")
    for label, value in summary_table(result):
        colour = ""
        if label == "gate":
            colour = "green" if value.startswith("PASS") else "red" if "FAIL" in value else ""
        if label == "drift":
            colour = "red" if value.startswith("RETRAIN") else "green" if "ok —" in value else ""
        table.add_row(label, f"[{colour}]{value}[/{colour}]" if colour else value)
    console.print(table)


def to_markdown(result: RetrainResult) -> str:
    """The run log: what ran, how long it took, and what it decided."""
    lines = [
        f"# Retrain run {result.started_at:%Y-%m-%dT%H:%M:%S}Z",
        "",
        f"- season: `{result.season}`",
        f"- split: `{result.split.label}` "
        f"(train <= {result.split.train_through}, val {result.split.val_season}, "
        f"test >= {result.split.test_from})",
        f"- dry run: `{result.dry_run}`",
        f"- wall clock: {result.seconds:.1f}s",
        "",
        "## Steps",
        "",
        "| step | status | seconds | detail |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {s.name} | {s.status} | {s.seconds:.1f} | {s.detail or ''} |" for s in result.steps
    ]
    lines += ["", "## Summary", "", "| | |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in summary_table(result)]

    if result.gate is not None:
        from nfl.models.train_ensemble import gate_table

        lines += ["", "## Promotion gate", ""]
        for note in result.gate.notes:
            lines += [f"> {note}", ""]
        table = gate_table(result.gate)
        lines += ["| " + " | ".join(table.columns) + " |"]
        lines += ["|" + "|".join(["---"] * len(table.columns)) + "|"]
        for row in table.iter_rows():
            cells = [f"{v:.6f}" if isinstance(v, float) else str(v) for v in row]
            lines += ["| " + " | ".join(cells) + " |"]
    return "\n".join(lines).rstrip() + "\n"


def report_path(result: RetrainResult, root: Path | None = None) -> Path:
    """``data/reports/retrain-<timestamp>.md``."""
    base = (config.DATA_DIR if root is None else Path(root)) / "reports"
    return base / f"retrain-{result.started_at:%Y%m%dT%H%M%S}.md"


def write_report(result: RetrainResult, root: Path | None = None) -> Path:
    """Write :func:`to_markdown` under ``data/reports/`` and return the path."""
    path = report_path(result, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(result), encoding="utf-8")
    return path


def steps_for(*, skip_ingest: bool = False) -> Sequence[str]:
    """The step names this run will attempt, in order."""
    if not skip_ingest:
        return RETRAIN_STEPS
    return tuple(s for s in RETRAIN_STEPS if s not in INGEST_STEPS)
