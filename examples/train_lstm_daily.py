"""Single-symbol PyTorch LSTM example using the collected Kiwoom daily database.

Install: uv sync --extra ml --inexact --no-install-project
Train:   uv run --no-sync python examples/train_lstm_daily.py --epochs 20
Predict: uv run --no-sync python examples/train_lstm_daily.py --checkpoint PATH
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_NAMES = ["open_log_return", "high_log_return", "low_log_return",
                 "close_log_return", "log_volume_change"]


def _require_legacy_database(connection):
    """Never bypass the approved sample index of a cleaned training database."""
    message = ("Cleaned training databases require the approved training_samples index; "
               "use examples/train_lstm30.py. This legacy daily LSTM example supports raw databases only.")
    objects = {row[0].casefold() for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    if "training_samples" in objects:
        raise ValueError(message)
    columns = {row[1].casefold() for row in connection.execute("PRAGMA table_info(daily_bars)")}
    if "segment_id" in columns:
        raise ValueError(message)
    if "metadata" in objects:
        for key, value in connection.execute(
                "SELECT key,value FROM metadata WHERE lower(key) IN ('schema_version','requires_training_samples')"):
            if key.casefold() == "requires_training_samples":
                # Presence is enough: an incomplete or contradictory marker
                # must not silently turn a cleaned database into a raw one.
                raise ValueError(message)
            try:
                version = json.loads(value)
            except (TypeError, ValueError):
                version = value
            if ((isinstance(version, str) and "clean-daily" in version.casefold())
                    or (isinstance(value, str) and "clean-daily" in value.casefold())):
                raise ValueError(message)


def load_bars(database: Path, symbol: str, exchange: str, start: str, end: str | None = None):
    """Read one raw symbol; reject cleaned DBs and never create missing files."""
    if not database.is_file():
        raise ValueError(f"Database not found: {database}")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        _require_legacy_database(connection)
        rows = connection.execute(
            "SELECT trade_date, open, high, low, close, volume FROM daily_bars "
            "WHERE symbol = ? AND exchange = ? AND trade_date >= ? "
            "AND (? IS NULL OR trade_date <= ?) ORDER BY trade_date",
            (symbol, exchange, start, end, end),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) < 2:
        raise ValueError(f"Not enough rows for {symbol}/{exchange}; check DB, exchange and start date.")
    return prepare_bars(rows)


def prepare_bars(rows):
    """Shared cleaning for DB rows and fresh API rows in the signal example."""
    if len(rows) < 2:
        raise ValueError("Not enough daily bars.")
    dates = np.array([row[0] for row in rows])
    values = np.array([row[1:] for row in rows], dtype=np.float64)
    # Suspended/invalid zero-price records are excluded. The target is the next valid bar.
    valid = (np.isfinite(values).all(axis=1)
             & (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0)
             & (values[:, 1] >= values[:, :4].max(axis=1))
             & (values[:, 2] <= values[:, :4].min(axis=1)))
    dropped = int((~valid).sum())
    if dropped:
        print(f"Excluded {dropped} invalid/zero-price bars.", file=sys.stderr, flush=True)
    if valid.sum() < 2:
        raise ValueError("Not enough valid OHLCV rows.")
    return dates[valid], values[valid], dropped


def make_features(bars: np.ndarray) -> np.ndarray:
    # Day t prices relative to day t-1 close, plus change in log(1 + volume).
    # Every input feature uses only the current or earlier completed bar.
    prices = np.log(bars[1:, :4] / bars[:-1, 3:4])
    volume = np.diff(np.log1p(bars[:, 4]))[:, None]
    return np.concatenate((prices, volume), axis=1).astype(np.float32)


def split_and_scale(features: np.ndarray, lookback: int):
    """Split by target date, then fit normalization on training inputs only."""
    targets = np.arange(lookback, len(features))
    if len(targets) < 20:
        raise ValueError("Need at least lookback + 21 valid daily bars for the 70/15/15 split.")
    train_end, val_end = int(len(targets) * 0.70), int(len(targets) * 0.85)
    splits = (targets[:train_end], targets[train_end:val_end], targets[val_end:])
    # A target at index j reads features[j-lookback:j], never features[j].
    last_train_target = int(splits[0][-1])
    fit_values = features[:last_train_target]
    mean = fit_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = fit_values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    normalized = ((features - mean) / std).astype(np.float32)
    return normalized, splits, mean, std


class DailyWindows(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 targets: np.ndarray, lookback: int):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.targets = targets
        self.lookback = lookback

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = int(self.targets[index])
        return self.features[target - self.lookback:target], self.labels[target]


class LSTMClassifier(nn.Module):
    """[batch, days, 5 features] -> LSTM -> last hidden state -> one logit."""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs)
        return self.head(hidden[-1]).squeeze(-1)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    loss_sum, count = 0.0, 0
    probabilities, labels = [], []
    with torch.set_grad_enabled(training):
        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; check input data and learning rate.")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            count += len(target)
            loss_sum += loss.item() * len(target)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(target.cpu().numpy())
    return loss_sum / count, np.concatenate(probabilities), np.concatenate(labels)


def classification_metrics(probabilities, labels):
    prediction, actual = probabilities >= 0.5, labels > 0.5
    tp = int((prediction & actual).sum())
    fp = int((prediction & ~actual).sum())
    fn = int((~prediction & actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": float((prediction == actual).mean()),
        "precision_up": precision,
        "recall_up": recall,
        "f1_up": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "actual_up_fraction": float(actual.mean()),
        "predicted_up_fraction": float(prediction.mean()),
    }


def latest_prediction(model, features, mean, std, lookback, dates, device):
    if len(features) < lookback:
        raise ValueError("Not enough rows for the checkpoint's lookback window.")
    values = ((features[-lookback:] - mean) / std).astype(np.float32)
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.from_numpy(values)[None].to(device))).item()
    return {
        "as_of_date": str(dates[-1]),
        "target": "next valid trading bar close > as_of_date close",
        "up_probability": probability,
        "threshold": 0.5,
        "predicted_direction": "UP" if probability >= 0.5 else "NOT_UP",
    }


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for prefix, name in (("train", "Train"), ("val", "Validation")):
        axes[0].plot(epochs, [r[f"{prefix}_loss"] for r in history], label=name)
        axes[1].plot(epochs, [r[f"{prefix}_accuracy"] for r in history], label=name)
    for axis, title in zip(axes, ("Binary cross-entropy", "Direction accuracy")):
        axis.set(xlabel="Epoch", title=title)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def train(args, device):
    database = args.db or Path("data/kiwoom_daily/domestic_daily.sqlite3")
    dates, bars, dropped = load_bars(database, args.symbol, args.exchange, args.start)
    features = make_features(bars)
    labels = (features[:, 3] > 0).astype(np.float32)
    normalized, splits, mean, std = split_and_scale(features, args.lookback)
    # Windows overlap naturally, but each split has disjoint, ordered target dates.
    loaders = [DataLoader(DailyWindows(normalized, labels, indices, args.lookback),
                          batch_size=args.batch_size, shuffle=(i == 0), num_workers=0)
               for i, indices in enumerate(splits)]
    architecture = {"input_size": len(FEATURE_NAMES), "hidden_size": args.hidden_size,
                    "num_layers": args.layers, "dropout": args.dropout}
    model = LSTMClassifier(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device={device}; valid bars={len(bars)}; windows={sum(map(len, splits))}; "
          f"train/val/test={tuple(map(len, splits))}; parameters={parameter_count:,}", flush=True)
    output = args.output_dir or Path("outputs/lstm") / (
        args.symbol + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_prob, train_y = run_epoch(model, loaders[0], device, optimizer)
        val_loss, val_prob, val_y = run_epoch(model, loaders[1], device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "train_accuracy": classification_metrics(train_prob, train_y)["accuracy"],
               "val_accuracy": classification_metrics(val_prob, val_y)["accuracy"]}
        history.append(row)
        print(f"Epoch {epoch:02d} | train loss={train_loss:.4f} "
              f"| val loss={val_loss:.4f} | val accuracy={row['val_accuracy']:.3f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping: validation loss stopped improving.", flush=True)
                break
    model.load_state_dict(best_state)
    # Test is evaluated only after all model/epoch selection has finished.
    test_loss, test_prob, test_y = run_epoch(model, loaders[2], device)
    scores = classification_metrics(test_prob, test_y)
    train_majority = int(labels[splits[0]].mean() >= 0.5)
    scores.update(loss=test_loss, train_majority_class=train_majority,
                  train_majority_baseline_accuracy=float((test_y == train_majority).mean()))
    target_dates = dates[1:]
    metadata = {
        "database": str(database.resolve()), "symbol": args.symbol, "exchange": args.exchange,
        "start": args.start, "lookback": args.lookback, "feature_names": FEATURE_NAMES,
        "valid_bars": len(bars), "excluded_bars": dropped, "trainable_parameters": parameter_count,
        "seed": args.seed, "torch_version": str(torch.__version__), "device": str(device),
        "architecture": architecture,
        "training": {"max_epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.lr, "patience": args.patience},
        "splits": {name: {"samples": len(indices), "first_target_date": str(target_dates[indices[0]]),
                          "last_target_date": str(target_dates[indices[-1]])}
                   for name, indices in zip(("train", "validation", "test"), splits)},
    }
    torch.save({"model_state_dict": best_state, "metadata": metadata,
                "mean": mean.tolist(), "std": std.tolist(), "best_epoch": best_epoch},
               output / "best_model.pt")
    latest = latest_prediction(model, features, mean, std, args.lookback, dates, device)
    save_json(output / "metrics.json", {"metadata": metadata, "best_epoch": best_epoch,
                                        "best_validation_loss": best_loss, "test": scores})
    save_json(output / "history.json", history)
    save_json(output / "latest_prediction.json", latest)
    save_json(output / "test_predictions.json", [
        {"input_end_date": str(target_dates[j - 1]), "target_date": str(target_dates[j]),
         "up_probability": float(prob), "actual_up": int(actual)}
        for j, prob, actual in zip(splits[2], test_prob, test_y)
    ])
    plot_history(history, output / "training_curve.png")
    print(f"Test accuracy={scores['accuracy']:.3f}; "
          f"train-majority baseline={scores['train_majority_baseline_accuracy']:.3f}", flush=True)
    print(json.dumps(latest, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output.resolve()}", flush=True)


def predict(args, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["metadata"]
    dates, bars, _ = load_bars(args.db or Path(config["database"]),
                              config["symbol"], config["exchange"], config["start"])
    model = LSTMClassifier(**config["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    latest = latest_prediction(model, make_features(bars),
                               np.array(checkpoint["mean"], dtype=np.float32),
                               np.array(checkpoint["std"], dtype=np.float32),
                               config["lookback"], dates, device)
    print(json.dumps(latest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--symbol", default="005930")
    parser.add_argument("--exchange", default="KRX")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Load a saved model and predict without training")
    args = parser.parse_args()
    for name in ("lookback", "hidden_size", "layers", "epochs", "batch_size", "patience", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.dropout < 1:
        parser.error("lr must be positive and finite; dropout must be in [0, 1)")
    try:
        date.fromisoformat(args.start)
    except ValueError:
        parser.error("start must be an ISO date (YYYY-MM-DD)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this PyTorch environment; use --device cpu")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    try:
        if args.checkpoint:
            predict(args, device)
        else:
            train(args, device)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
