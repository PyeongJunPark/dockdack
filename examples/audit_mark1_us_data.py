"""Read-only US daily-data diagnosis; no fitting, broker access, or source edits.

Audits stored raw/clean OHLCV aggregates and 30+1 cached windows independently.
The original HTTP candle payload is not persisted in the raw DB, so exact
raw-to-clean copying alone cannot prove the provider's historical correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVES = ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA')
FIELDS = ('symbol', 'exchange', 'trade_date', 'open', 'high', 'low', 'close', 'volume',
          'trade_value', 'change', 'change_rate', 'adjustment_type', 'adjustment_rate',
          'currency', 'collected_at')
SPLITS = (
    ('NVDA', '2024-06-03', '2024-06-14', '2024-06-07', '2024-06-10', 10,
     'https://www.sec.gov/Archives/edgar/data/1045810/000104581024000144/nvda-20240607.htm'),
    ('AAPL', '2020-08-24', '2020-09-04', '2020-08-28', '2020-08-31', 4,
     'https://www.apple.com/newsroom/2020/07/apple-reports-third-quarter-results/'),
    ('TSLA', '2020-08-24', '2020-09-04', '2020-08-28', '2020-08-31', 5, None),
    ('TSLA', '2022-08-22', '2022-08-30', '2022-08-24', '2022-08-25', 3,
     'https://ir.tesla.com/press-release/tesla-announces-three-one-stock-split'),
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def connection(path):
    result = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    result.row_factory = sqlite3.Row
    result.execute('PRAGMA query_only=ON')
    result.execute('BEGIN')
    return result


def statistics(db):
    query = '''SELECT COUNT(*) AS rows,MIN(trade_date) AS first_date,MAX(trade_date) AS last_date,
        SUM(currency != 'USD' OR currency IS NULL) AS non_usd,
        SUM(open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
            OR CAST(open AS REAL)<=0 OR CAST(high AS REAL)<=0
            OR CAST(low AS REAL)<=0 OR CAST(close AS REAL)<=0) AS invalid_nonpositive_prices,
        SUM(CAST(high AS REAL)<MAX(CAST(open AS REAL),CAST(low AS REAL),CAST(close AS REAL))
            OR CAST(low AS REAL)>MIN(CAST(open AS REAL),CAST(high AS REAL),CAST(close AS REAL))) AS invalid_ohlc_bounds,
        SUM(volume IS NULL OR typeof(volume)!='integer' OR volume<0) AS invalid_volume,
        SUM(volume=0) AS zero_volume,MIN(CAST(close AS REAL)) AS minimum_close,
        MAX(CAST(close AS REAL)) AS maximum_close,
        SUM(adjustment_type IS NOT NULL) AS adjustment_tag_rows,
        SUM(CAST(high AS REAL)/CAST(low AS REAL)>=2) AS intraday_range_at_least_2x
        FROM daily_bars'''
    return dict(db.execute(query).fetchone())


def prevalence(take, stop, mask):
    t, s = take[mask], stop[mask]
    counts = {'take_only': int((t & ~s).sum()), 'stop_only': int((s & ~t).sum()),
              'both_touch': int((t & s).sum()), 'neither': int((~t & ~s).sum())}
    n = len(t)
    return {'count': n, 'class_counts': counts,
            'class_rates': {name: count / n if n else None for name, count in counts.items()},
            'any_take_rate': float(t.mean()) if n else None,
            'any_stop_rate': float(s.mean()) if n else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', type=Path, default=Path('C:/Users/user/Desktop/dockdack-data-collection/data/kiwoom_daily/clean-20260916-v1/us_daily_clean.sqlite3'))
    parser.add_argument('--cache', type=Path, default=ROOT / 'outputs/mark1/cache/us-7bc645867ce3c050.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/mark1/us-diagnostic-20260916/data-audit.json')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output; overwrite forbidden')
    started = time.monotonic()
    clean = connection(args.clean)
    metadata = {key: json.loads(value) for key, value in clean.execute('SELECT key,value FROM metadata')}
    source = Path(metadata['source_database'])
    raw = connection(source)
    protected = [source, args.clean, args.cache] + [ROOT / name for name in (
        'dockdack/kiwoom.py', 'dockdack/daily_dataset.py', 'dockdack/clean_daily_dataset.py',
        'dockdack/mark1_data.py', 'dockdack/mark1_selective_features.py')]
    protected += [Path(str(path) + '-wal') for path in (source, args.clean) if Path(str(path) + '-wal').exists()]
    before = {str(path.resolve()): digest(path) for path in protected}
    errors = []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    report = {'scope': 'Stored-data and label diagnosis only; not a strategy change or proof of future predictability',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'errors': errors,
              'source_database': str(source), 'clean_database': str(args.clean), 'cache': str(args.cache),
              'metadata': metadata, 'source_http_candle_payload_retained': False,
              'decoder': {'price_parser': 'Decimal string, strips comma/sign via absolute=True; no scale factor',
                          'request_adjusted': True, 'request_exchange_conversion': False,
                          'currency': 'USD', 'saved_raw_scope': 'Decoded bars; original per-candle HTTP response is not persisted'},
              'statistics': {}, 'representatives': {}, 'corporate_actions': [],
              'limitations': ['Raw-to-clean equality does not independently verify provider prices.',
                             'Current catalog survivor bias and target observed-positive-volume selection remain.',
                             'Corporate-action checks are bounded examples, not exhaustive event reconciliation.',
                             'Large changes may be real market events, reverse splits or data errors; flags alone do not prove corruption.']}
    try:
        for label, db in (('raw', raw), ('clean', clean)):
            print(f'AUDIT {label} full OHLCV aggregates', flush=True)
            report['statistics'][label] = statistics(db)
            report['statistics'][label]['schemas'] = {table: [dict(row) for row in db.execute(f'PRAGMA table_info({table})')]
                                                       for table in ('daily_bars', 'instruments')}
        check(report['statistics']['clean']['non_usd'] == 0, 'clean non-USD rows')
        for field in ('invalid_nonpositive_prices', 'invalid_ohlc_bounds', 'invalid_volume'):
            check(report['statistics']['clean'][field] == 0, 'clean ' + field)
        report['source_matches_clean_creation_hash'] = before[str(source.resolve())] == metadata['source_fingerprints'][source.name]['sha256']
        check(report['source_matches_clean_creation_hash'], 'raw DB differs from frozen clean source')
        sessions = {day: ordinal for day, ordinal in clean.execute('SELECT session_date,ordinal FROM sessions')}
        for symbol in REPRESENTATIVES:
            info = {}
            for label, db in (('raw', raw), ('clean', clean)):
                info[label] = dict(db.execute('''SELECT COUNT(*) AS rows, MIN(trade_date) AS first_date,
                    MAX(trade_date) AS last_date, MIN(CAST(close AS REAL)) AS minimum_close,
                    MAX(CAST(close AS REAL)) AS maximum_close FROM daily_bars WHERE symbol=? AND exchange='ND' ''', (symbol,)).fetchone())
                info[label]['latest'] = [dict(row) for row in db.execute('''SELECT trade_date,open,high,low,close,volume,
                    trade_value,currency,collected_at FROM daily_bars WHERE symbol=? AND exchange='ND'
                    ORDER BY trade_date DESC LIMIT 3''', (symbol,))]
            info['approved_windows'] = clean.execute("SELECT COUNT(*) FROM training_samples WHERE symbol=? AND exchange='ND'", (symbol,)).fetchone()[0]
            report['representatives'][symbol] = info
        for symbol, first, last, pre, post, factor, reference in SPLITS:
            rows = [dict(row) for row in raw.execute('''SELECT trade_date,open,high,low,close,volume,trade_value,
                change,change_rate,adjustment_type,adjustment_rate FROM daily_bars
                WHERE symbol=? AND exchange='ND' AND trade_date BETWEEN ? AND ? ORDER BY trade_date''', (symbol, first, last))]
            lookup = {row['trade_date']: row for row in rows}
            deltas = []
            for previous, current in zip(rows, rows[1:]):
                actual = Decimal(current['close']) - Decimal(previous['close'])
                stored = Decimal(current['change']) if current['change'] is not None else None
                deltas.append({'date': current['trade_date'], 'computed_adjusted_close_delta': float(actual),
                               'stored_change': float(stored) if stored is not None else None,
                               'stored_over_adjusted_delta': float(stored / actual) if stored is not None and actual else None,
                               'computed_close_return_pct': float((Decimal(current['close']) / Decimal(previous['close']) - 1) * 100),
                               'stored_change_rate': current['change_rate']})
            ratios = [float(Decimal(row['trade_value']) / (Decimal(row['close']) * Decimal(row['volume'])))
                      for row in rows if row['trade_value'] is not None and row['volume'] > 0]
            report['corporate_actions'].append({'symbol': symbol, 'ex_date': post, 'split_factor': factor,
                'official_reference': reference, 'rows': rows, 'adjusted_close_deltas': deltas,
                'post_open_over_pre_close': float(Decimal(lookup[post]['open']) / Decimal(lookup[pre]['close'])),
                'turnover_over_adjusted_close_times_volume_range': [min(ratios), max(ratios)],
                'clean_rows_exact_raw_copy': all(clean.execute('SELECT open,high,low,close,volume FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date=?',
                    (symbol, 'ND', row['trade_date'])).fetchone() is not None and tuple(clean.execute('SELECT open,high,low,close,volume FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date=?',
                    (symbol, 'ND', row['trade_date'])).fetchone()) == tuple(row[key] for key in ('open', 'high', 'low', 'close', 'volume')) for row in rows)})
        print('AUDIT cache labels, gaps and approved historical spans', flush=True)
        with np.load(args.cache, allow_pickle=False) as archive:
            dates, ids, starts = archive['target_dates'], archive['symbol_ids'], archive['starts']
            bars, ohlc = archive['bars'], archive['target_ohlc']
            manifest, cache_config = json.loads(str(archive['manifest'].item())), json.loads(str(archive['cache_config'].item()))
        check(cache_config['database_sha256'] == before[str(args.clean.resolve())], 'cache source hash mismatch')
        check(np.all(starts >= 0) and np.all(starts + 29 < len(bars)), 'cache starts out of range')
        check(np.isfinite(ohlc).all() and np.all(ohlc > 0), 'cache target prices invalid')
        check(np.all(ohlc[:, 1] >= ohlc.max(axis=1)) and np.all(ohlc[:, 2] <= ohlc.min(axis=1)), 'cache target OHLC bounds')
        take = ohlc[:, 1] / ohlc[:, 0] >= 1.01 * (1 - 1e-12)
        stop = ohlc[:, 2] / ohlc[:, 0] <= .991 * (1 + 1e-12)
        yearly = {}
        years = dates.astype('datetime64[D]').astype('datetime64[Y]').astype(int) + 1970
        for year in np.unique(years):
            yearly[str(year)] = prevalence(take, stop, years == year)
        periods = {}
        for label, begin, end in (('train2012_2019', '2012-01-01', '2019-12-31'), ('train2014_2021', '2014-01-01', '2021-12-31')):
            periods[label] = prevalence(take, stop, (dates >= np.datetime64(begin).astype(int)) & (dates <= np.datetime64(end).astype(int)))
        previous_close = bars[starts + 29, 3].astype(np.float64)
        gap = np.log(ohlc[:, 0]) - np.log(previous_close)
        extreme = np.argsort(np.abs(gap))[-32:]
        sampled = np.random.default_rng(20260916).choice(len(dates), size=384, replace=False)
        positive = np.flatnonzero(take & ~stop)
        negative = np.flatnonzero(~(take & ~stop))
        sampled = np.unique(np.r_[sampled, extreme, positive[np.linspace(0, len(positive) - 1, 32).astype(int)],
                                  negative[np.linspace(0, len(negative) - 1, 32).astype(int)]])
        symbol_metadata = {row['symbol_id']: row for row in manifest['symbols']}
        for symbol in REPRESENTATIVES:
            selected_ids = [sid for sid, row in symbol_metadata.items() if row['symbol'] == symbol and row['exchange'] == 'ND']
            indices = np.flatnonzero(np.isin(ids, selected_ids))
            if len(indices):
                sampled = np.unique(np.r_[sampled, indices[np.linspace(0, len(indices) - 1, 16).astype(int)]])
            report['representatives'][symbol]['cache_events'] = len(indices)
            report['representatives'][symbol]['cache_event_prevalence'] = prevalence(take, stop, np.isin(ids, selected_ids))
        report['cache'] = {'path': str(args.cache), 'bars': len(bars), 'events': len(dates),
            'symbols': len(symbol_metadata), 'yearly': yearly, 'periods_before_fold_universe_filter': periods,
            'first_target': str(np.datetime64(int(dates.min()), 'D')), 'last_target': str(np.datetime64(int(dates.max()), 'D')),
            'gap_count_abs_log_ge_log1_5': int((np.abs(gap) >= np.log(1.5)).sum()),
            'gap_count_open_le_half_previous_close': int((gap <= np.log(.5)).sum()),
            'gap_count_open_ge_double_previous_close': int((gap >= np.log(2)).sum()),
            'gap_count_abs_log_ge_log10': int((np.abs(gap) >= np.log(10)).sum()),
            'gap_quantiles': dict(zip(('min', 'p001', 'p01', 'median', 'p99', 'p999', 'max'),
                                    map(float, np.quantile(gap, [0, .001, .01, .5, .99, .999, 1])))),
            'extreme_examples': [{'sample_index': int(i), 'symbol': symbol_metadata[int(ids[i])]['symbol'],
                'exchange': symbol_metadata[int(ids[i])]['exchange'], 'target_date': str(np.datetime64(int(dates[i]), 'D')),
                'entry_open': float(ohlc[i, 0]), 'previous_close': float(previous_close[i]),
                'open_over_previous_close': float(np.exp(gap[i])), 'success': bool(take[i] and not stop[i])} for i in extreme[-16:]]}
        mismatches = Counter()
        checked_bars = 0
        sampled_records = []
        columns = ','.join('"' + field + '"' for field in FIELDS)
        for i in sampled:
            info = symbol_metadata[int(ids[i])]
            symbol, exchange = info['symbol'], info['exchange']
            day = str(np.datetime64(int(dates[i]), 'D'))
            sample = clean.execute('SELECT * FROM training_samples WHERE symbol=? AND exchange=? AND target_date=?', (symbol, exchange, day)).fetchone()
            if sample is None:
                mismatches['missing_approved_sample'] += 1
                continue
            rows = list(clean.execute(f'SELECT {columns},segment_id,source_rowid,quality_flags FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date',
                                      (symbol, exchange, sample['input_start_date'], day)))
            if len(rows) != 31:
                mismatches['wrong_31_span'] += 1
                continue
            ordinals = [sessions[row['trade_date']] for row in rows]
            if not np.all(np.diff(ordinals) == 1):
                mismatches['session_gap'] += 1
            if any(row['segment_id'] != sample['segment_id'] for row in rows):
                mismatches['wrong_segment'] += 1
            if rows[-2]['trade_date'] != sample['input_end_date'] or rows[-1]['volume'] <= 0:
                mismatches['wrong_endpoint_or_inactive_target'] += 1
            expected_history = np.asarray([[float(row[key]) for key in ('open', 'high', 'low', 'close', 'volume')] for row in rows[:-1]], dtype=np.float32)
            if not np.array_equal(expected_history, bars[starts[i]:starts[i] + 30]):
                mismatches['cache_history'] += 1
            target = np.asarray([float(rows[-1][key]) for key in ('open', 'high', 'low', 'close')])
            if not np.array_equal(target, ohlc[i]):
                mismatches['cache_target'] += 1
            entry_d, high_d, low_d = (Decimal(rows[-1][key]) for key in ('open', 'high', 'low'))
            decimal_success = high_d / entry_d >= Decimal('1.01') * (1 - Decimal('1e-12')) and low_d / entry_d > Decimal('.991') * (1 + Decimal('1e-12'))
            if decimal_success != bool(take[i] and not stop[i]):
                mismatches['independent_decimal_label'] += 1
            rowids = [row['source_rowid'] for row in rows]
            originals = {row['rowid']: row for row in raw.execute(f'SELECT rowid,{columns} FROM daily_bars WHERE rowid IN ({",".join("?" for _ in rowids)})', rowids)}
            for row in rows:
                original = originals.get(row['source_rowid'])
                if original is None or any(row[key] != original[key] for key in FIELDS):
                    mismatches['raw_clean_exact_fields'] += 1
            checked_bars += len(rows)
            if i in set(extreme.tolist()):
                sampled_records.append({'sample_index': int(i), 'symbol': symbol, 'date': day,
                    'target_adjustment_type': rows[-1]['adjustment_type'], 'target_adjustment_rate': rows[-1]['adjustment_rate'],
                    'target_quality_flags': rows[-1]['quality_flags'], 'entry_open': target[0], 'prior_close': float(previous_close[i])})
        report['window_audit'] = {'sample_count': len(sampled), 'checked_31bar_rows': checked_bars,
                                'random_seed': 20260916, 'random_count': 384, 'selection': 'uniform events + extremes + spread positive/negative labels + 7 representatives',
                                'mismatches': dict(mismatches), 'extreme_target_metadata': sampled_records}
        check(not mismatches, 'sampled approved raw/clean/cache/label mismatch')
        report['unused_change_fields_finding'] = 'Stored change/change_rate are not consistently restated for future splits; inspected examples exhibit split-factor-scale mismatch. Current models compute price returns from OHLC and do not read these fields.'
        report['interpretation'] = 'Structural checks do not prove historical price correctness. No all-zero-success-label, constant100xUSprice scaling, sampled raw-to-clean copying, or sampled chronological-alignment bug was found. Material split-related price discontinuities survive in approved windows; see corporate-action-audit.json for specific evidence and exposure counts. Their causal contribution to broad zero-signal behavior is unmeasured.'
    finally:
        clean.close()
        raw.close()
    after = {str(path.resolve()): digest(path) for path in protected}
    report['protected_sha256_before'], report['protected_sha256_after'] = before, after
    report['protected_files_unchanged'] = before == after
    check(before == after, 'protected files changed during diagnosis')
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    report['passed_structural_checks'] = not errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({'output': str(args.output), 'passed_structural_checks': report['passed_structural_checks'],
                      'errors': errors, 'window_audit': report['window_audit'],
                      'statistics': {key: {k: v for k, v in value.items() if k != 'schemas'} for key, value in report['statistics'].items()},
                      'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False), flush=True)
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
