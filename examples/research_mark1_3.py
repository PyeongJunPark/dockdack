"""Run the separate Mark1.3 daily pre-open research prototype.

Reads the raw local daily SQLite database only. Results are hypothetical
open-to-close proxies, never deployment approval or an order signal.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import torch

from dockdack.clean_daily_dataset import load_sessions
from dockdack.mark1_3_research import (chronological_splits, load_raw_daily_candidates,
                                       run_experiment)
from dockdack.research_artifacts import write_new_json


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market', choices=('domestic', 'us'), required=True)
    parser.add_argument('--database', type=Path, help='Raw local daily SQLite database (read only)')
    parser.add_argument('--start', default='2017-01-01')
    parser.add_argument('--train-end', default='2021-12-31')
    parser.add_argument('--validation-end', default='2022-12-31')
    parser.add_argument('--test-end', default='2024-12-31')
    parser.add_argument('--max-symbols', type=int, default=8)
    parser.add_argument('--train-cap', type=int, default=20_000)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--cost-bps', type=float, default=20)
    parser.add_argument('--cost-grid-bps', type=float, nargs='+', default=(0, 10, 20, 40, 80),
                        help='Declared roundtrip cost sensitivity; signal policy stays fixed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New child folder under outputs/mark1 (ignored by Git)')
    args = parser.parse_args(argv)
    destination = args.output_dir.resolve()
    output_root = (ROOT / 'outputs' / 'mark1').resolve()
    if not destination.is_relative_to(output_root) or destination == output_root or destination.exists():
        parser.error('Choose a new child folder under outputs/mark1')
    database = args.database or ROOT / 'data' / 'kiwoom_daily' / f'{args.market}_daily.sqlite3'
    if not database.is_file():
        parser.error(f'Raw database not found: {database}')
    try:
        sessions, calendar_info = load_sessions(args.market, date.fromisoformat(args.start),
                                                date.fromisoformat(args.test_end) + timedelta(days=1))
        samples = load_raw_daily_candidates(
            database, args.market, start=args.start, train_end=args.train_end,
            test_end=args.test_end, max_symbols=args.max_symbols, seed=args.seed,
            session_dates=sessions)
        splits = chronological_splits(samples, train_end=args.train_end,
                                      validation_end=args.validation_end, test_end=args.test_end,
                                      sessions=tuple(sessions))
        report, checkpoint = run_experiment(samples, splits, cost_bps=args.cost_bps,
                                            seed=args.seed, epochs=args.epochs,
                                            train_cap=args.train_cap,
                                            cost_grid_bps=args.cost_grid_bps)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    report['calendar'] = calendar_info
    report['model_artifact'] = 'research_model.pt'
    destination.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, destination / 'research_model.pt')
    write_new_json(destination / 'report.json', report)
    print(f'Mark1.3 research report: {destination / "report.json"}')
    print('Research only; no profitability or deployment claim.')


if __name__ == '__main__':
    main()
