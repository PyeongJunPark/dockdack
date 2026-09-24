"""Bounded read-only supplement: split discontinuities and frozen feature rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.audit_mark1_us_data import ROOT, FIELDS, connection, digest
from examples.train_mark1_selective import selective_splits, feature_array


CASES = (
    ('FCEL', 'ND', '2024-11-11', '2024-11-06', '2024-11-15',
     'https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2024-540', 'Official 1-for-30 split-adjusted trading Nov11; extreme one-session scale spike then Nov12 adjustment marker.'),
    ('BNED', 'NY', '2024-06-12', '2024-06-07', '2024-06-17',
     'https://www.sec.gov/Archives/edgar/data/1634117/000119312524159120/d818124dex991.htm', 'Official 1-for-100 reverse split Jun12; historical/current units discontinuous.'),
    ('BBSI', 'ND', '2024-06-24', '2024-06-18', '2024-06-28',
     'https://ir.bbsi.com/news-events/press-releases/detail/373/bbsi-declares-four-for-one-stock-split', 'Official 4-for-1 post-split trading Jun24; historical/current units discontinuous.'),
    ('SONY', 'NY', '2024-09-30', '2024-09-25', '2024-10-10',
     'https://www.sec.gov/Archives/edgar/data/313838/000110465924060803/tm2414258d1_6k.htm', 'Suspicious 5x scale jump Sep30; US ADR timing differs from Japanese common-share split. Do not assert Sep30 is official ADR ex-date.'),
)


def main():
    destination = ROOT / 'outputs/mark1/us-diagnostic-20260916/corporate-action-audit.json'
    if destination.exists():
        raise FileExistsError(destination)
    base_path = destination.with_name('data-audit.json')
    base = json.loads(base_path.read_text(encoding='utf-8'))
    cache_path = Path(base['cache']['path'])
    db_paths = [Path(key) for key in base['protected_sha256_before'] if key.endswith('.sqlite3')]
    clean_path = next(path for path in db_paths if 'clean' in path.name)
    raw_path = next(path for path in db_paths if 'clean' not in path.name)
    feature_paths = [ROOT / f'outputs/mark1/selective-20260916/us/{fold}/features/audit.npy'
                     for fold in ('walk_2022', 'walk_2024')]
    protected = [raw_path, clean_path, cache_path, ROOT / 'dockdack/mark1_selective_features.py'] + feature_paths
    before = {str(path.resolve()): digest(path) for path in protected}
    clean, raw = connection(clean_path), connection(raw_path)
    cache = np.load(cache_path, allow_pickle=False)
    dataset = SimpleNamespace(**{key: cache[key] for key in ('bars', 'starts', 'target_dates', 'symbol_ids', 'target_ohlc')})
    manifest = json.loads(str(cache['manifest']))
    symbols = {row['symbol_id']: row for row in manifest['symbols']}
    sessions = np.asarray([np.datetime64(row[0], 'D').astype(np.int64)
                           for row in clean.execute('SELECT session_date FROM sessions ORDER BY ordinal')], dtype=np.int64)
    bars = dataset.bars
    query_gaps = np.log(dataset.target_ohlc[:, 0]) - np.log(bars[dataset.starts + 29, 3].astype(np.float64))
    historical_gaps = np.zeros(len(bars), dtype=np.float64)
    historical_gaps[1:] = np.log(bars[1:, 0].astype(np.float64)) - np.log(bars[:-1, 3].astype(np.float64))
    # Each validated history is a contiguous single-symbol segment; boundary
    # gaps between packed segments never fall inside its 29 comparisons.
    history_counts = {}
    for factor in (1.5, 2):
        prefix = np.r_[0, np.cumsum(np.abs(historical_gaps) > np.log(factor), dtype=np.int64)]
        history_counts[factor] = prefix[dataset.starts + 30] - prefix[dataset.starts + 1]
    report = {'scope': 'Read-only descriptive data diagnosis; no retraining or adjusted re-evaluation',
              'cases': [], 'folds': {}, 'feature_cache_alignment': {}, 'errors': []}
    splits_by_fold = {}
    for fold in ('walk_2022', 'walk_2024'):
        print('CORPORATE audit ' + fold, flush=True)
        splits = selective_splits(dataset, sessions, fold)
        splits_by_fold[fold] = splits
        contract_path = ROOT / f'outputs/mark1/selective-20260916/us/{fold}/features/features.json'
        contract = json.loads(contract_path.read_text())
        for part, indices in splits.items():
            actual = hashlib.sha256(indices.tobytes()).hexdigest()
            if actual != contract['index_sha256'][part]:
                report['errors'].append(f'{fold}/{part}: index hash mismatch')
        report['folds'][fold] = {}
        for part in ('train', 'tune', 'probability_calibration', 'policy_calibration', 'audit'):
            indices = splits[part]
            entry = {'count': len(indices), 'maximum_abs_log_query_gap': float(np.abs(query_gaps[indices]).max())}
            for factor in (1.5, 2):
                query_mask = np.abs(query_gaps[indices]) > np.log(factor)
                history_mask = history_counts[factor][indices] > 0
                entry[str(factor)] = {'query_gap_count': int(query_mask.sum()),
                    'history_29_gap_window_count': int(history_mask.sum()),
                    'either_history_or_query_count': int((query_mask | history_mask).sum()),
                    'either_fraction': float((query_mask | history_mask).mean())}
            report['folds'][fold][part] = entry
        indices = splits['audit']
        saved = np.load(ROOT / f'outputs/mark1/selective-20260916/us/{fold}/features/audit.npy', mmap_mode='r')
        positions = np.random.default_rng(20260916).choice(len(indices), 64, replace=False)
        fresh = feature_array(dataset, indices[positions])
        selected = np.asarray(saved[positions])
        exact = bool(np.array_equal(fresh, selected))
        report['feature_cache_alignment'][fold] = {'sampled_rows': 64, 'columns': fresh.shape[1],
            'index_sha256_verified_all_partitions': not any(fold in error for error in report['errors']),
            'bitwise_equal': exact, 'maximum_absolute_difference': float(np.abs(fresh - selected).max())}
        if not exact:
            report['errors'].append(f'{fold}: regenerated features mismatch')
    audit_indices = splits_by_fold['walk_2024']['audit']
    for symbol, exchange, day, start, end, source, note in CASES:
        rows = list(clean.execute('SELECT * FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date',
                                  (symbol, exchange, start, end)))
        case = {'symbol': symbol, 'exchange': exchange, 'event_date': day, 'official_source': source,
                'interpretation': note, 'raw_clean_all_fields_equal': True, 'rows': []}
        for row in rows:
            original = raw.execute('SELECT * FROM daily_bars WHERE rowid=?', (row['source_rowid'],)).fetchone()
            equal = original is not None and all(original[key] == row[key] for key in FIELDS)
            case['raw_clean_all_fields_equal'] &= equal
            item = {key: row[key] for key in ('trade_date', 'open', 'high', 'low', 'close', 'volume', 'trade_value', 'adjustment_type', 'adjustment_rate', 'quality_flags', 'segment_id')}
            item['raw_clean_equal'] = equal
            item['turnover_div_close_volume'] = (float(row['trade_value']) / (float(row['close']) * row['volume'])
                if row['trade_value'] and row['volume'] else None)
            case['rows'].append(item)
        sid = next(sid for sid, info in symbols.items() if info['symbol'] == symbol and info['exchange'] == exchange)
        event_day = int(np.datetime64(day, 'D').astype(np.int64))
        event = np.flatnonzero((dataset.symbol_ids == sid) & (dataset.target_dates == event_day))
        case['event_cache_indices'] = event.tolist()
        if len(event):
            event = int(event[0])
            case['entry_over_prior_close'] = float(np.exp(query_gaps[event]))
            event_bar = dataset.starts[event] + 30
            # Include the split-day query and all future windows containing
            # both sides of its discontinuity, not isolated adjusted candles.
            affected = ((dataset.symbol_ids[audit_indices] == sid)
                & (((dataset.starts[audit_indices] < event_bar)
                    & (dataset.starts[audit_indices] + 29 >= event_bar))
                   | (audit_indices == event)))
            case['walk_2024_audit_crossing_gap_windows_including_event_query'] = int(affected.sum())
            if symbol == 'FCEL':
                # The Nov12 return to the surrounding unit scale is a second
                # discontinuity. Its windows overlap the Nov11 windows.
                second = np.flatnonzero((dataset.symbol_ids == sid)
                    & (dataset.target_dates == int(np.datetime64('2024-11-12', 'D').astype(np.int64))))
                second_event = int(second[0])
                second_bar = dataset.starts[second_event] + 30
                second_affected = ((dataset.symbol_ids[audit_indices] == sid)
                    & (((dataset.starts[audit_indices] < second_bar)
                        & (dataset.starts[audit_indices] + 29 >= second_bar))
                       | (audit_indices == second_event)))
                case['two_discontinuity_dates'] = ['2024-11-11', '2024-11-12']
                case['two_discontinuities_union_approved_window_count'] = int((affected | second_affected).sum())
        report['cases'].append(case)
    extremes = audit_indices[np.argsort(np.abs(query_gaps[audit_indices]))[-32:][::-1]]
    report['walk_2024_extreme_query_gaps'] = [{'sample_index': int(index),
        'symbol': symbols[int(dataset.symbol_ids[index])]['symbol'],
        'date': str(np.datetime64(int(dataset.target_dates[index]), 'D')),
        'open_over_prior_close': float(np.exp(query_gaps[index]))} for index in extremes]
    raw.close()
    clean.close()
    after = {str(path.resolve()): digest(path) for path in protected}
    report.update({'protected_sha256_before': before, 'protected_sha256_after': after,
        'protected_files_unchanged': before == after,
        'interpretation': 'There are real split-related scale discontinuities in input features despite structurally valid OHLC and exact raw-to-clean copying. Gap screens are descriptive, not proof every flagged move is erroneous. No repair or retraining was done, so their causal contribution to overall abstention remains unknown.'})
    if before != after:
        report['errors'].append('Protected source hashes changed')
    report['checks_passed'] = not report['errors']
    destination.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(destination), 'folds': report['folds'], 'features': report['feature_cache_alignment'],
        'cases': [{key: value for key, value in case.items() if key != 'rows'} for case in report['cases']],
        'checks_passed': report['checks_passed'], 'protected_files_unchanged': before == after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
