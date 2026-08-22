# NFL Win-Probability Forecasting

Given a snapshot of an NFL game at any moment (target: late Q4), emit
`P(home wins)`, `P(away wins) = 1 - P(home)`, and a confidence score.

**Hard rule:** betting lines and Kalshi prices are *never* model inputs. They are
used only during evaluation, to measure edge. See
[`docs/ADR/0001-no-market-inputs.md`](docs/ADR/0001-no-market-inputs.md).

## Quick start

```bash
uv sync --extra gpu --dev   # drop --extra gpu for a CPU-only install
uv run pytest
uv run nfl --help
```

GPU users: call `nfl.gpu.preload_cuda_libraries()` before importing TensorFlow,
otherwise TF cannot `dlopen` the pip-installed CUDA libraries and falls back to
CPU. See the GPU notes in [`CLAUDE.md`](CLAUDE.md).

## Commands

| Command | Purpose |
|---|---|
| `uv run nfl ingest` | Pull nflverse / weather / injury / news data into Parquet + DuckDB |
| `uv run nfl train` | Train a model on season-based splits |
| `uv run nfl predict GAME_ID [--at TIME]` | Predict win probability and write a ledger row |
| `uv run nfl evaluate` | Log-loss / Brier / calibration, and edge vs market |

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline overview
- [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) — exhaustive entity/attribute spec
- [`docs/ADR/`](docs/ADR/) — architecture decision records
- [`CLAUDE.md`](CLAUDE.md) — conventions for agents working in this repo
