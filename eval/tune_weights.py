"""
Runs the synthetic_transforms eval under the baseline fusion weights plus a
handful of alternative configs, reports which performs best by mean
recall@1 across transform types, and patches the winning weights directly
into the shipped webapp/public/data/model_params.json (channels.<name>.weight)
so the webapp ships whatever won here rather than requiring a manual step.
"""
import json
import os
import subprocess
import sys

WEBAPP_MODEL_PARAMS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "webapp", "public", "data", "model_params.json",
)

CANDIDATES = {
    "baseline": {"dino_color": 0.5, "dino_grey": 0.2, "clip": 0.2, "fourier": 0.1},
    "dino_heavy": {"dino_color": 0.65, "dino_grey": 0.15, "clip": 0.15, "fourier": 0.05},
    "balanced_shape": {"dino_color": 0.4, "dino_grey": 0.25, "clip": 0.15, "fourier": 0.2},
}

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(name, weights):
    out_path = os.path.join(HERE, "synthetic_transforms", f"results_{name}.json")
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "synthetic_transforms", "run.py"),
            "--weights", json.dumps(weights),
            "--out", out_path,
            "--n-queries", "150",
        ],
        check=True,
    )
    with open(out_path) as f:
        return json.load(f)


def mean_recall1(result):
    r1s = [v["recall@1"] for v in result["results"].values()]
    return sum(r1s) / len(r1s)


def main():
    summary = {}
    for name, weights in CANDIDATES.items():
        print(f"=== running config: {name} {weights} ===")
        result = run_one(name, weights)
        summary[name] = {"weights": weights, "mean_recall@1": mean_recall1(result), "per_transform": result["results"]}

    best = max(summary, key=lambda k: summary[k]["mean_recall@1"])
    print("\nsummary:")
    for name, s in summary.items():
        print(f"  {name}: mean recall@1 = {s['mean_recall@1']:.4f}")
    print(f"\nbest config: {best} ({summary[best]['weights']})")

    with open(os.path.join(HERE, "weight_tuning_summary.json"), "w") as f:
        json.dump({"candidates": summary, "best": best}, f, indent=1)

    with open(WEBAPP_MODEL_PARAMS) as f:
        model_params = json.load(f)
    for channel, weight in summary[best]["weights"].items():
        model_params["channels"][channel]["weight"] = weight
    with open(WEBAPP_MODEL_PARAMS, "w") as f:
        json.dump(model_params, f)
    print(f"patched {WEBAPP_MODEL_PARAMS} channels[*].weight with the '{best}' config")


if __name__ == "__main__":
    main()
