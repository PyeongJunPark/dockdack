"""Run using an isolated wheel interpreter, from outside the checkout."""
import argparse
import json
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
    results = []
    for model in MODEL_IDS:
        client = PrototypeProcessClient(model, timeout=60)
        try:
            health = client.request("health")
            for market in ("domestic", "us"):
                metadata = client.request("metadata", market=market)
                prediction = client.request("predict", market=market,
                    bars=[[100., 103., 99., 101., 10000.] for _ in range(30)], current_price="100")
                results.append({"model": model, "market": market, "worker": health["pid"],
                                "metadata_verified": bool(metadata), "prediction": prediction})
        finally:
            client.close()
    print(json.dumps({"installed": str(installed), "app_home": str(app_home()),
                      "no_accounts_or_orders": True, "results": results}, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
