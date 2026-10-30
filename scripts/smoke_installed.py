"""Smoke installed workers outside the checkout; pre-open models load metadata only."""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    import dockdack
    installed = Path(dockdack.__file__).resolve()
    if installed.is_relative_to(checkout):
        raise RuntimeError("Smoke must import the installed wheel, not the checkout")
    os.environ["DOCKDACK_MODEL_ROOT"] = str(args.model_root.resolve(strict=True))
    from dockdack.runtime_paths import checkout_root, app_home
    if checkout_root() is not None:
        raise RuntimeError("Installed runtime unexpectedly found a source checkout")
    from dockdack.v00_app import V00Window  # GUI imports, never starts an app/account.
    from dockdack.prototype_external import PrototypeProcessClient, MODEL_IDS
    from dockdack.signals.mark1_intraday_trigger import MODEL_IDS as DAILY_PROXY_IDS
    from dockdack.signals.mark1_target_horizon_trigger import MODEL_IDS as TARGET_HORIZON_IDS
    from dockdack.signals.preopen_series import PREOPEN_MODELS
    results = []
    for model in MODEL_IDS:
        client = PrototypeProcessClient(model, timeout=60)
        try:
            health = client.request("health")
            for market in ("domestic", "us"):
                metadata = client.request("metadata", market=market)
                # The pre-open worker deliberately has no bare predict RPC:
                # it requires a valid exchange session and 100 ranked inputs.
                preopen = model == "mark1-4-prototype" or model in PREOPEN_MODELS
                prediction = None if preopen else client.request(
                    "predict", market=market,
                    bars=[[100., 103., 99., 101., 10000.] for _ in range(30)], current_price="100")
                if model in DAILY_PROXY_IDS:
                    score = prediction.get("probability_success")
                    if (metadata.get("intraday_path_verified") is not False
                            or metadata.get("deployment_allowed") is not False
                            or prediction.get("strategy_id") != model
                            or prediction.get("score_scope") != "daily_open_whole_session_proxy_not_intraday"
                            or isinstance(score, bool) or not isinstance(score, (int, float))
                            or not math.isfinite(score) or not 0 <= score <= 1):
                        raise RuntimeError(f"Installed daily-proxy model contract failed: {model}/{market}")
                if model in TARGET_HORIZON_IDS:
                    score = prediction.get("probability_success")
                    if (metadata.get("intraday_path_verified") is not False
                            or metadata.get("deployment_allowed") is not False
                            or prediction.get("strategy_id") != model
                            or prediction.get("score_scope") != "next_open_proxy_not_intraday_verified"
                            or prediction.get("stop_loss_pct", False) is not None
                            or isinstance(score, bool) or not isinstance(score, (int, float))
                            or not math.isfinite(score) or not 0 <= score <= 1):
                        raise RuntimeError(f"Installed target/horizon model contract failed: {model}/{market}")
                results.append({"model": model, "market": market, "worker": health["pid"],
                                "metadata_verified": bool(metadata),
                                "prediction_checked": not preopen, "prediction": prediction})
        finally:
            client.close()
    print(json.dumps({"installed": str(installed), "app_home": str(app_home()),
                      "no_accounts_or_orders": True, "models": len(MODEL_IDS),
                      "markets": 2, "results": results}, ensure_ascii=False,
                     default=str, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
