# Data and research adversarial audit

Date: 2026-09-24. Checkout: `C:/Users/user/Desktop/dockdack`, branch `codex/optimization-adversarial-audit`, based on `6e3ace7`.

Scope: read-only review of collection, quality/indexing, research training, inference artifacts, backtests, exporters, reports, and local research storage. No production source, database, model, account state, or Git ref changed. No real broker requests, orders, model fitting, full-dataset rebuilds, or expensive benchmark reruns. This report is the sole intended written artifact. Python imports may create ordinary ignored bytecode caches.

## Executive result

Four actionable findings were established, including in-memory reproductions of malformed collection and mixed adjustment histories. Historical research reproduction also fails after directory consolidation and because one frozen source was converted to CRLF. These failures do **not** mean the saved portable prototype weights have changed or all existing prices are corrupt.

The three portable bundle manifest receipts pass. Their declarations correctly remain `research_only=true`, `research_qualified=false`, `deployment_allowed=false`, and `intraday_path_verified=false`. The reports explicitly acknowledge reused evaluation, losses/no signals, and unresolved corporate actions. Do not replace those qualifications with a success claim based on passing unit tests or numerical export parity.

## Confirmed findings

### R1 — P2: physical paths are part of immutable research identity; relocation breaks reproduction

Evidence:

- `examples/train_mark1.py:60-81`: `dataset_cache()` resolves the physical database path, includes it in `cache_config`, hashes the whole config into the cache filename, and reads/hashes that exact physical source even on a cache hit.
- `examples/backtest_mark1_deep.py:107-118`: `load_frozen_cache()` uses the original `source_contract['database_path']` without a relocation mapping. The half-percent and selective pipelines depend on this loader; `examples/train_mark1_0504.py:76-77` is one concrete caller.
- `examples/train_mark1_deep.py:314` and `examples/train_mark1_selective.py:315`: default database directory is the now-removed `dockdack-data-collection` sibling.
- `examples/export_mark1_0504.py:82-86,109`: supplied cache/training paths must equal the frozen absolute locations; the exporter also opens the original absolute DB path. Simply supplying a new path does not repair all consumers.
- `examples/report_mark1_0504.py:102-104`: a provenance hash is looked up using the current absolute summary path as its dictionary key. The stored key still names the removed `dockdack-mark_1` worktree.
- `examples/audit_mark1_us_corporate.py:33-40`: the supplementary audit follows absolute paths inside its old diagnostic input.
- `docs/dataset-cleaning.md:95` states that the cleaned databases are still in the removed sibling; `docs/mark1-selective-protocol.md:78,97` likewise has old dependency paths.

Bounded actual-artifact reproduction:

1. Read each `outputs/mark1/selective-20260916/{domestic,us}/source.json`.
2. Verify `Path(source['database_path']).is_file()` is false and canonical `data/kiwoom_daily/clean-20260916-v1/{market}_daily_clean.sqlite3` exists.
3. Invoke `load_frozen_cache(source, market, Path('outputs/mark1/cache'))`: both markets immediately raise `FileNotFoundError` for the removed sibling, before training or loading a large array.
4. Invoke `examples.report_mark1_0504.load_inputs()` with the three canonical directories: it raises `Completed training/backtest provenance mismatch`. Both old-path hash values exactly equal the current training-summary SHA-256; only the dictionary lookup key differs.
5. The deep and selective report input readers, which use relative layout for this step, accept their existing canonical artifacts. No report renderer was run or output overwritten.

Impact: successful model preservation does not imply reproducible training/backtesting/export/report commands. A user can be tempted to rewrite sealed metadata or regenerate expensive caches solely because a folder moved.

Remedy: add a new versioned, read-only artifact resolver that separates immutable logical source identity from physical location. Require exact content SHA-256 and expected market/schema before accepting a canonical relocation. Keep historical source contracts and cache keys unchanged; record the relocation outside the frozen record. Provide a new research entry point and explicit data/cache/run overrides. Do not weaken integrity checks or silently change historical hashes. Test relocation with an identical small fixture plus rejection of a different-byte fixture. Update current-location documentation separately from historical logs.

Effort: medium, about 1–2 days including cross-pipeline contract tests. Disposition: **retain** existing DBs/caches/manifests; **add** compatibility tooling, no deletion.

### R2 — P2: one hash-sealed deep-model source changes bytes on Windows checkout

Evidence:

- `examples/backtest_mark1_deep.py:95-103` checks the exact bytes of `dockdack/mark1_deep_models.py` against the frozen experiment.
- `.gitattributes` has preservation rules for other frozen deep/feature/training modules, but none for `dockdack/mark1_deep_models.py`.
- Existing deep protocol expected SHA-256: `50ae22f70278a9cd4ca0fb276d20260a2421c9874ac8f35846a10529d15d16d7`.
- Current on-disk CRLF bytes: 15,652 bytes, SHA-256 `d6ac2b375ae5c8078ce5770e5ff24550a2715e0b9e0b1eb067ed7cc4e07d847d`.
- Converting CRLF to LF **in memory only** gives 15,350 bytes and exactly the frozen hash. This is a newline conversion, not evidence of a changed mathematical implementation.

Reproduction: `verify_training_artifacts(Path('outputs/mark1/deep-20260916'), market)` fails for both markets with `Frozen training code changed: dockdack/mark1_deep_models.py`.

Cross-check: selective training code hashes match 10/10 files; half-percent training hashes match 13/13; deep matches 6/7.

Impact: historical deep backtest verification is blocked on the canonical checkout and can re-break on another Windows checkout. It is separate from R1 and must be fixed even after path relocation is supported.

Remedy: restore the **original recorded LF bytes**, add explicit byte-preservation attributes, and add a fixture/receipt check covering every source named by every frozen protocol. Do not rewrite the frozen hash to match the converted file. Validate in a disposable Windows checkout, not only the current working tree.

Effort: small, hours. Disposition: **retain** file and frozen receipts; byte restoration plus Git-attribute guard when implementation is authorized.

### R3 — P1: incremental raw dataset refresh can combine different adjusted-price bases

Evidence:

- `dockdack/daily_dataset.py:393` carries the previously completed latest date into refresh.
- `dockdack/daily_dataset.py:463-479` requests adjusted prices (`upd_stkpc_tp=1`).
- `dockdack/daily_dataset.py:512-523` stops pagination as soon as any bar reaches the old latest date, although the API still has older pages.
- `dockdack/daily_dataset.py:258-270` overwrites the returned overlap, with no comparison/revision generation or historical resynchronization.
- Contrast: the separate GUI `HistoryCache` does compare finalized overlapping bars and requests a full window on revision (`dockdack/history_cache.py:82-94`). That protection is absent from the persistent raw dataset collector.

Offline reproduction used only fake broker/store objects. Previously stored Sep21/Sep22/Sep23 closes were 100. The broker's new adjusted page contained Sep24/Sep23 closes of 50, with an older continuation page also containing adjusted closes of 50. Calling `_collect_instrument(..., known_latest='2026-09-23')` made one request, marked completion, and left:

```text
2026-09-21: 100
2026-09-22: 100
2026-09-23: 50
2026-09-24: 50
status: complete
```

Impact: a routine refresh after a split or a provider historical correction can create artificial return/gap/volatility features while every individual OHLC row remains valid. The quality cleaner only flags large interday changes, intentionally not deleting legitimate moves (`dataset_quality.py:245-246`), so it cannot establish a coherent adjustment basis. This is a demonstrated code risk, **not an attribution of every known existing corporate-action anomaly to this collector**.

Remedy: stage a new per-symbol adjusted-history generation. Detect changes in finalized overlaps and correction/adjustment metadata; if changed, fully refetch that symbol and atomically publish a coherent generation, retaining previous provenance. Provide a deliberate full-history refresh option. Do not rescale old rows heuristically or remove large returns based on outcomes. Test interrupted resync, unchanged overlap, split/reverse split, changed provider history, and old latest date on the first page.

Effort: medium, 1–3 days with collection/resume tests. Disposition: **retain** raw and cleaned v1; create a versioned corrected generation only after independent review and authorization.

### R4 — P2: malformed or wrong-symbol daily responses can be marked complete

Evidence:

- `dockdack/daily_dataset.py:495` defaults a missing required chart-list key to an empty list.
- `dockdack/daily_dataset.py:499-510` silently drops non-object/missing-date records and does not validate a returned symbol against the request.
- `dockdack/daily_dataset.py:280` marks a last page complete; `:389-391` skips completed symbols on later normal collection runs.
- The separate `dockdack/history.py:88-93,94-100` already rejects a returned-symbol mismatch, missing list, malformed records, or invalid dates.

Offline reproduction used the actual collector and parser with fake broker/store only:

```text
body {'return_code': 0, 'return_msg': 'OK'}
=> save_page(requested='005930', bars=0, has_next=False), no error

body {'stk_cd': '999999', 'stk_dt_pole_chart_qry': [one valid row]}
requested symbol '005930'
=> save_page(requested='005930', bars=1, has_next=False), no error
```

Impact: a schema/API regression can look like legitimate no-history completion and therefore stop retries. A mismatched response can be persisted under the requested identity and survive later structural cleaning. An explicit correctly shaped empty list may still be a legitimate zero-history response; the missing-key case is different.

Remedy: validate response envelope/list and any available market/symbol/exchange identity before saving any page; reject malformed elements instead of silently discarding them. Share a pure validator between history and dataset paths where their field contracts really agree. Save an error/retryable progress state, never completion, on invalid payloads. Add collection-level tests for missing key, wrong identity, partially malformed list, and legitimate explicit empty list.

Effort: small to medium, half a day to one day. Disposition: **retain** collector; improve validation, not delete it.

## Optimization and maintainability opportunities (not measured speedup claims)

### O1 — P2 performance: sample caps do not cap packed host memory; feature generation allocates whole folds

- `dockdack/mark1_data.py:210-218,243-247,258-265,294-295`: all approved retained bars/target arrays are accumulated and concatenated; caps are applied to split index arrays, not packed storage. Lowering `--max-train-samples` does not proportionally lower this memory cost.
- `examples/train_mark1.py:73-80`: NPZ members are materialized into memory; this is not an mmap bank. `dataset_cache()` hashes each full cleaned DB before and after a cache hit (`:65,:81`), so even cached startup scans gigabytes.
- `examples/train_mark1_selective.py:80-91,125-132` and `examples/train_mark1_0504.py:105-117,175-183`: allocate an entire float32 feature matrix before saving, then reopen it with mmap. Batch-size only caps intermediate feature computation, not the output allocation.
- `mark1_selective_models.py:273-276` hashes the same large train/tune arrays again for each architecture/seed request. This is useful integrity protection but repeated work.

Measured metadata, without loading array payloads:

| Current artifact | Shape/size |
|---|---|
| domestic packed bars | 3,454,129 × 5 float32 |
| domestic starts / target OHLC | 2,668,151 starts; same count × 4 float64 OHLC |
| US packed bars | 8,869,839 × 5 float32 |
| US starts / target OHLC | 8,088,544 starts; same count × 4 float64 OHLC |
| two raw cache NPZ files | 774,315,016 bytes total |
| selective feature files | 20 files; 6,534,365,856 bytes |
| half-percent feature files | 20 files; 6,532,818,048 bytes |
| largest individual feature file | 1,104,000,128 bytes (1.5M × 184 × float32 plus header) |

Remedy for a new version: packed `.npy` mmap store plus immutable manifest, direct chunked `open_memmap` feature writes with atomic completion, preflight host/GPU memory estimate, and content-addressed shared feature cache. Reuse verified hashes only within an explicitly immutable snapshot/session; do not blindly trust mtime alone or remove source-change checks. Measure peak RSS, wall time, and cold/warm I/O before promising gains. Keep full uncapped arrays when the walk-forward/evaluation contract needs them, but distinguish that cost from training caps in CLI help.

Effort: medium to large, 2–5 days plus parity tests. Existing sealed caches are **retain**, not an immediate 13 GB deletion opportunity.

### O2 — P2 reproducibility: research dependency installation is hidden outside declared extras

`examples/train_mark1_selective.py:322` unconditionally imports both CatBoost and LightGBM. `pyproject.toml:22-34` declares Torch/NumPy/Matplotlib for `ml` and CatBoost for `prototype`, but no LightGBM extra. Current local `outputs/mark1/selective-deps` masks this on this machine. The runtime bundle and training dependency sets are intentionally different, but a clean installation has no explicit complete selective-research environment contract.

Remedy: a dedicated versioned research extra/lock/environment recipe, separate from the small GUI/prototype dependency set. State CPU/GPU backend and platform versions; a seed alone (`train_mark1.py:138-143`) is not a promise of cross-driver bitwise-identical GPU training. Test import-only/research `--help` in a fresh isolated installation, then a tiny fixture fit when separately authorized. Keep the local dependency directory until the replacement environment is verified.

Effort: small to medium. Main packaging auditor owns implementation decisions.

### O3 — P3 maintenance: deliberate frozen copies should not be casually deduplicated

- `mark1_backtest.py` and `mark1_0504_backtest.py` are near-copy accounting engines; the latter explicitly explains frozen-version isolation at lines1-11.
- Several training/export/report files repeat JSON/hash/source-contract checks. Report checks have diverged: newer reports cross-check completion/ledgers more thoroughly than early `report_mark1.py` and `report_mark1_backtest.py`.
- `scripts/report_lstm30_training.py` and the early model comparison reporters are historical experiment tools, not substitutes for current portfolio/strategy-lot P&L.

Remedy: keep frozen v1 modules as archival compatibility code. For new experiments, extract a new tested portfolio core with explicit target/cost/path semantics and thin versioned adapters; add cross-engine golden parity fixtures before adoption. Add a registry of artifact families, logical source IDs, required source hashes, and report readers rather than many scattered absolute-path assumptions. Do not refactor a source whose bytes are in a saved bundle's runtime contract in place.

Effort: medium; lower priority than R1–R4.

## Research limitations that are already disclosed (not newly discovered bugs)

1. Current exported CatBoost prototypes were trained on actual OPEN entries, with no price augmentation in the selective/half-percent experiments (`train_mark1_selective.PROTOCOL`, `reports/mark1-0504-20260920/REPORT.md:20`). Earlier sequence/deep experiments did use counterfactual prices. A current-price query is accepted by inference, but not a newly validated remaining-session first-touch probability. `mark1_prototype_inference.py:33-55` and `mark1_0504_inference.py:35-59` are explicit. Better research requires entry-time/intraday labels and execution-aware validation, not merely deeper layers or changing barriers.
2. Repeated 2025+ evaluation is correctly called reused history, not a new untouched holdout. Train/tune/calibration/policy/audit folds have calendar purging and train-availability universe restrictions. These reduce specific leakage but do not erase repeated model-selection exposure, survivorship, or corporate-action uncertainty.
3. Clean v1 requires a next-session observed positive-volume target; that availability selection remains. The catalog is current, not point-in-time. Corporate-action discontinuities are still acknowledged; half-percent training quarantines FCEL/BNED/BBSI, not a corrected whole-market DB.
4. Clean `daily_bars` is the union of approved sample spans, not a complete post-entry execution-price universe. Backtests lock capital on missing bars, mark stale prices, and flag uncertain gap/terminal liquidations rather than claiming real fills. See `mark1_backtest.py:83-90` and `report_mark1_backtest.py:150-153`. A future full valuation panel should be separate from the signal eligibility panel.
5. Fee scenarios are idealized constant round-trip bps, not a verified account-specific fee schedule. Current reports distinguish no-signal precision from 0% precision, and no-trade 0% portfolio return from success. Preserve that behavior.
6. Export numerical equivalence covers a small deterministic set of histories/prices; it is not calibration, profitability, or deployment evidence. Existing risk flags are appropriate.

## Retain / move / delete guidance

| Item | Recommendation | Reason / condition |
|---|---|---|
| Raw and cleaned DBs, summary JSON, dataset release/checksum manifests | Retain | Raw provenance and approved sample-index contracts; do not edit v1 in place |
| `models/mark1_prototype`, `models/mark1_0504`, `models/mark1_1_prototype` | Retain all three | Named prototype and original half-percent provenance are intentionally distinct; apparent weight duplication is not permission to delete the source family |
| Old `models/mark1`, `models/lstm30` and research baselines | Retain until dependency registry proves unused | Current comparison/export code protects and compares earlier checkpoints |
| Frozen training runs, selections, calibrated predictions, reports, source hashes | Retain | Reproduce negative findings and prevent selective reporting; essential evidence, not clutter |
| `outputs/mark1/*/features/*.npy` | Retain now; potential managed cache later | ~13.07 GB logical size, but regeneration is expensive and frozen consumers verify these bytes |
| `outputs/mark1/selective-deps`, `prototype-gui-deps` | Retain now; move only through a tested environment migration | Currently used runtime/research dependencies, not ordinary plot output |
| `resume.pt`, CatBoost snapshots, failed partial artifacts | Archive candidate after inventory and recovery policy | Can be large; only prune after confirming final artifacts/hashes and that resumability is no longer required |
| Python `__pycache__`, regenerated test PNG/logs not referenced by evidence | Disposable candidates, not deleted in this audit | Rebuildable; must distinguish evidence screenshots/logs from temporary smoke output |
| Research source duplicates | Retain frozen copies; future versioned refactor | Exact-byte receipts make blind deduplication unsafe |
| Recovery Git bundle and migration receipts | Retain | Rollback/audit evidence, small compared with dataset/features |

No specific DB, model, or experiment directory is approved for deletion by this review.

Machine-readable reproduction outcomes and review coverage: `outputs/optimization-audit/research-repro-results.json`. This contains observations, not executable account or training instructions.

## Coverage and exclusions

Focused source/contract review covered:

- Data: `daily_dataset.py`, `clean_daily_dataset.py`, `dataset_identity.py`, `dataset_quality.py`, `history.py`, `history_cache.py`, `local_data_paths.py`; `scripts/package_daily_snapshot.py`, `scripts/warm_history_cache.py`.
- Core ML/data: `ml30.py`, `mark1_data.py`, `mark1_backtest_data.py`, `mark1_deep_data.py`, `mark1_0504_data.py`; causal-feature/model-contract portions of `mark1_models.py`, `mark1_deep_models.py`, `mark1_selective_features.py`, `mark1_selective_models.py`, `mark1_metrics.py`, `mark1_deep_validation.py`, `mark1_selective_policy.py`.
- Inference: loading/provenance/risk contracts in `mark1_inference.py`, `mark1_deep_inference.py`, `mark1_selective_inference.py`, `mark1_prototype_inference.py`, `mark1_0504_inference.py`, `mark1_1_prototype_inference.py`; adapter/trigger boundaries were inspected for ownership but runtime order behavior is assigned to the engine auditor.
- Training/backtests/export: `examples/train_lstm_daily.py` legacy boundary, `train_lstm30.py` loader/indexing, `train_mark1.py`, `train_mark1_deep.py`, `train_mark1_selective.py`, `train_mark1_0504.py`; four `backtest_mark1*` runners and the two daily portfolio engines; three `export_mark1*` tools.
- Reports/audits: all five `report_mark1*` families' relevant loading/configuration/provenance sections, `scripts/report_lstm30_training.py`; US data/training/corporate/probability diagnostics and deep audit entrypoints, with more detailed review of corporate/training checks.
- Evidence: training protocols/code hash registries, portable manifest receipts, cleaned dataset docs/manifest context, current saved report conclusions, actual NPZ headers and feature-file sizes. Relevant unit-test source was inspected for collector/history coverage gaps; no full test suite was run by this sub-agent (root owns global tests).

This was a risk-focused review, not a claim that every line in every listed file has been exhaustively proven correct. Unreviewed or deliberately excluded: GUI rendering, account/order engine and strategy-lot accounting, broker credentials/live responses, platform API specification revalidation, complete corporate-action/security-master reconciliation, exhaustive database scans, training convergence, real fill simulation, clean-machine package installation, GPU performance benchmarking, and all external network research.

## Suggested implementation order after approval

1. Repair exact-byte checkout preservation and add an all-frozen-contract preflight (R2).
2. Add content-verified relocation tooling and fix new-current-path documentation (R1), without mutating historical manifests.
3. Harden collection envelope/identity validation (R4), then design atomic coherent adjusted-history refresh (R3).
4. Make research environment/commands reproducible (O2).
5. Profile host memory/I/O; implement a versioned chunked/mmap cache only if measured bottlenecks justify it (O1).
6. Pursue corrected corporate-action data and truly entry-time-aware validation before interpreting new threshold/depth experiments as a path to reliable intraday buying. No training or promotion is authorized by this audit.

---

[종합 보고서로 돌아가기](README.md)
