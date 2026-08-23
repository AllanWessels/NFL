"""Rich rendering for :class:`~nfl.data.live.base.LiveSnapshot`.

Kept out of the adapters so neither of them imports ``rich``, and out of
``cli.py`` so the layout can be tested without a terminal.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nfl.data.live.base import LiveSnapshot, seconds_to_clock

__all__ = ["render_snapshot", "snapshot_tables"]

_STATE_STYLE = {"pre": "cyan", "in": "bold green", "post": "dim"}


def _score_table(snapshot: LiveSnapshot) -> Table:
    table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    table.add_column("")
    table.add_column("team")
    table.add_column("score", justify="right")
    table.add_column("record")
    table.add_column("TO", justify="right")
    for side, team, score, record, timeouts in (
        ("away", snapshot.away_team, snapshot.away_score, snapshot.away_record, snapshot.away_timeouts),
        ("home", snapshot.home_team, snapshot.home_score, snapshot.home_record, snapshot.home_timeouts),
    ):
        has_ball = snapshot.possession is not None and snapshot.possession == team
        table.add_row(
            "\N{AMERICAN FOOTBALL}" if has_ball else " ",
            Text(team, style="bold" if has_ball else ""),
            str(score),
            (record.overall if record else "") or "",
            "-" if timeouts is None else str(timeouts),
        )
    return table


def _situation_table(snapshot: LiveSnapshot) -> Table:
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column("field", style="dim")
    table.add_column("value")

    period = "OT" if snapshot.is_overtime else f"Q{snapshot.period}"
    table.add_row("clock", f"{period} {seconds_to_clock(snapshot.clock_seconds)}")
    if snapshot.down is not None:
        distance = snapshot.distance if snapshot.distance is not None else "?"
        line = f"{_ordinal(snapshot.down)} & {distance}"
        if snapshot.yards_to_endzone is not None:
            line += f", {snapshot.yards_to_endzone} to the end zone"
        table.add_row("down", line)
    elif snapshot.yards_to_endzone is not None:
        table.add_row("ball on", f"{snapshot.yards_to_endzone} to the end zone")
    if snapshot.possession:
        table.add_row("possession", snapshot.possession)
    if snapshot.is_red_zone:
        table.add_row("", Text("red zone", style="red"))
    table.add_row("secs left (reg)", str(snapshot.game_seconds_remaining))
    if not snapshot.timeouts_exact:
        table.add_row("", Text("timeouts are the feed's current value, not as-of", style="yellow"))

    table.add_row("plays", str(snapshot.play_count))
    table.add_row("drives", str(len(snapshot.drives)))
    wallclock = snapshot.last_play_wallclock
    if wallclock is not None:
        age = datetime.now(tz=UTC) - wallclock
        minutes = age.total_seconds() / 60
        style = "yellow" if snapshot.is_stale() else "dim"
        table.add_row("last play", Text(f"{wallclock:%H:%M:%SZ} ({minutes:,.0f} min ago)", style=style))
    if snapshot.venue and snapshot.venue.name:
        detail = snapshot.venue.name
        if snapshot.venue.roof:
            detail += f" ({snapshot.venue.roof})"
        table.add_row("venue", detail)
    if snapshot.weather and snapshot.weather.display_value:
        table.add_row("weather", snapshot.weather.display_value)
    return table


def _drive_table(snapshot: LiveSnapshot, limit: int = 5) -> Table:
    table = Table(title="recent drives", title_style="bold", box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("team")
    table.add_column("result")
    table.add_column("detail", style="dim")
    for drive in snapshot.drives[-limit:]:
        table.add_row(
            str(drive.drive_number or ""),
            drive.team or "",
            drive.result or drive.display_result or "",
            drive.description or "",
        )
    return table


def _injury_table(snapshot: LiveSnapshot, limit: int = 8) -> Table:
    table = Table(title="injuries", title_style="bold", box=None, pad_edge=False)
    table.add_column("team")
    table.add_column("player")
    table.add_column("pos", style="dim")
    table.add_column("status")
    for injury in snapshot.injuries[:limit]:
        table.add_row(injury.team, injury.player_name, injury.position or "", injury.status or "")
    return table


def snapshot_tables(snapshot: LiveSnapshot) -> Panel:
    """The whole snapshot as one renderable."""
    title = f"{snapshot.away_team} @ {snapshot.home_team}"
    if snapshot.game_id:
        title += f"  [dim]{snapshot.game_id}[/dim]"
    subtitle = snapshot.status_detail or snapshot.state
    parts: list[RenderableType] = [_score_table(snapshot), Text(""), _situation_table(snapshot)]
    if snapshot.drives:
        parts += [Text(""), _drive_table(snapshot)]
    if snapshot.injuries:
        parts += [Text(""), _injury_table(snapshot)]
    return Panel(
        Group(*parts),
        title=title,
        subtitle=f"[{_STATE_STYLE.get(snapshot.state, 'white')}]{subtitle}[/] · via {snapshot.source}",
        border_style=_STATE_STYLE.get(snapshot.state, "white"),
    )


def render_snapshot(snapshot: LiveSnapshot, console: Console | None = None) -> None:
    (console or Console()).print(snapshot_tables(snapshot))


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(n, f"{n}th")
