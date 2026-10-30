"""Run separate Mark1.4 random neural rule evolution on read-only daily data.

This is a historical research program, not a trading signal producer. It does
not import a broker, GUI, or order service. Opening and closing fills are
hypothetical, and model artifacts are explicitly marked non-deployable.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import platform
import time

import numpy as np
import torch

from dockdack.clean_daily_dataset import load_sessions, source_fingerprints
from dockdack.mark1_4_data import load_mark14_candidates, mark14_chronological_splits
from dockdack.mark1_4_evolution import (
    evolve, genome_to_payload as v1_genome_to_payload, GENOME_SIZE,
)
from dockdack.mark1_4_sparse import evolve_sparse, genome_to_payload as v2_genome_to_payload
from dockdack.research_artifacts import write_new_json


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('v1', 'v2'), default='v1',
                        help='v1 absolute score gate or v2 train-frozen sparse coverage search')
    parser.add_argument('--market', choices=('domestic', 'us'), required=True)
    parser.add_argument('--database', type=Path, help='Local raw daily SQLite database, opened read-only')
    parser.add_argument('--start', default='2017-01-01')
    parser.add_argument('--calibration-end', default='2017-12-31')
    parser.add_argument('--train-start', default='2018-01-01')
    parser.add_argument('--train-end', default='2021-12-31')
    parser.add_argument('--validation-end', default='2022-12-31')
    parser.add_argument('--test-end', default='2024-12-31')
    parser.add_argument('--max-symbols', type=int, default=100)
    parser.add_argument('--population-size', type=int, default=64)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--chunk-rows', type=int, default=2048)
    parser.add_argument('--cost-bps', type=float, default=20.0)
    parser.add_argument('--initial-equity', type=float,
                        help='Per-market hypothetical starting balance; default 10m KRW/10k USD')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='A new child folder under ignored outputs/mark1')
    args = parser.parse_args(argv)

    destination = args.output_dir.resolve()
    output_root = (ROOT / 'outputs' / 'mark1').resolve()
    if not destination.is_relative_to(output_root) or destination == output_root or destination.exists():
        parser.error('Choose a new, unused child folder under outputs/mark1')
    database = (args.database or ROOT / 'data' / 'kiwoom_daily'
                / f'{args.market}_daily.sqlite3').resolve()
    if not database.is_file():
        parser.error(f'Raw database not found: {database}')
    try:
        first, calibration_end, train_start, train_end, validation_end, test_end = (
            date.fromisoformat(value) for value in (
                args.start, args.calibration_end, args.train_start, args.train_end,
                args.validation_end, args.test_end))
        if not first < calibration_end < train_start <= train_end < validation_end < test_end:
            raise ValueError('Expected start < calibration_end < train_start <= train_end < validation_end < test_end')
        started = time.perf_counter()
        before = source_fingerprints(database)
        sessions, calendar_info = load_sessions(args.market, first, test_end + timedelta(days=1))
        samples = load_mark14_candidates(
            database, args.market, start=args.start, calibration_end=args.calibration_end,
            train_end=args.train_end, test_end=args.test_end,
            max_symbols=args.max_symbols, session_dates=sessions)
        after = source_fingerprints(database)
        if before != after:
            raise ValueError('Raw database changed during candidate loading; rerun from a stable snapshot')
        data_ready = time.perf_counter()
        splits = mark14_chronological_splits(
            samples, sessions, train_start=args.train_start, train_end=args.train_end,
            validation_end=args.validation_end, test_end=args.test_end)
        initial_equity = (args.initial_equity if args.initial_equity is not None else
                          (10_000_000 if args.market == 'domestic' else 10_000))
        training = evolve if args.variant == 'v1' else evolve_sparse
        report, champion = training(
            samples, splits, seed=args.seed, population_size=args.population_size,
            generations=args.generations, device=args.device, chunk_rows=args.chunk_rows,
            cost_bps=args.cost_bps, allocation=0.1, max_positions=10,
            initial_equity=initial_equity)
        evolution_ready = time.perf_counter()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    report['calendar'] = calendar_info
    report['raw_source_fingerprints'] = before
    report['requested_variant'] = args.variant
    report['model_artifact'] = 'champion.json' if champion is not None else None
    report['named_model_artifact'] = 'champion.npz' if champion is not None else None
    report['runtime'] = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'cuda_device': torch.cuda.get_device_name() if report['search']['device'] == 'cuda' else None,
        'candidate_loading_seconds': round(data_ready - started, 3),
        'evolution_and_evaluation_seconds': round(evolution_ready - data_ready, 3),
    }
    destination.mkdir(parents=True, exist_ok=False)
    if champion is not None:
        artifact = {
            'model': report['experiment'],
            'research_only': True,
            'deployment_allowed': False,
            'genome_size': GENOME_SIZE,
            'genome': champion.tolist(),
            'source_report': 'report.json',
        }
        if args.variant == 'v2':
            threshold = report['selected_strategy']['frozen_numeric_score_threshold']
            artifact['frozen_train_numeric_threshold'] = threshold
            artifact['threshold_provenance'] = report['selected_strategy']['threshold_provenance']
            payload = v2_genome_to_payload(champion, threshold)
        else:
            payload = v1_genome_to_payload(champion)
        write_new_json(destination / 'champion.json', artifact)
        with (destination / 'champion.npz').open('xb') as stream:
            np.savez_compressed(stream, **payload)
    write_new_json(destination / 'report.json', report)
    print(f'Mark1.4 {args.variant} research report: {destination / "report.json"}', flush=True)
    print('Research only; no profitability, real fill, or deployment claim.', flush=True)


if __name__ == '__main__':
    main()
