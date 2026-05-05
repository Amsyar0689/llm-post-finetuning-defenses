#!/usr/bin/env python
"""Generate summary figures from evaluation JSON files in results/."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ATTACK_PATTERN = re.compile(
    r"^attack_success_ratio(?P<ratio>\d+)_lr(?P<lr>[0-9eE\.-]+)_ep(?P<ep>\d+)\.json$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures") / "results")
    return parser.parse_args()


def normalize_lr(raw_lr: str) -> str:
    value = raw_lr.lower().replace("e-5", "e-05")
    if value.startswith("2e-5"):
        return "2e-05"
    if value.startswith("5e-5"):
        return "5e-05"
    if value.startswith("1e-5"):
        return "1e-05"
    return value


def load_rows(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        records = data.get("records", [])
        if path.name == "baseline_eval.json" or data.get("fine_tuned") is False:
            unsafe_count = int(data.get("unsafe_count", 0))
            safe_count = int(data.get("refusal_count", 0))
            attack_success_rate = float(data.get("attack_success_rate", 0.0))
            label = "baseline"
            ratio = None
            lr = None
            ep = None
        else:
            match = ATTACK_PATTERN.match(path.name)
            if not match:
                continue
            ratio = int(match.group("ratio"))
            lr = normalize_lr(match.group("lr"))
            ep = int(match.group("ep"))
            unsafe_count = int(data.get("unsafe_count", 0))
            safe_count = int(data.get("safe_count", 0))
            attack_success_rate = float(data.get("attack_success_rate", 0.0))
            label = f"r{ratio} / lr{lr} / ep{ep}"

        response_lengths = [len(str(record.get("response", ""))) for record in records]
        unsafe_records = sum(1 for record in records if record.get("judge_label") == "unsafe")
        refused_records = sum(1 for record in records if record.get("refused") is True)

        rows.append(
            {
                "file": path.name,
                "label": label,
                "ratio": ratio,
                "lr": lr,
                "ep": ep,
                "unsafe_count": unsafe_count,
                "safe_count": safe_count,
                "attack_success_rate": attack_success_rate,
                "num_records": len(records),
                "unsafe_records": unsafe_records,
                "refused_records": refused_records,
                "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0.0,
                "median_response_length": float(pd.Series(response_lengths).median()) if response_lengths else 0.0,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No JSON files found in {results_dir}")
    return frame


def sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["ratio_sort"] = ordered["ratio"].fillna(0).astype(int)
    ordered["lr_sort"] = ordered["lr"].fillna("0").map(
        lambda value: float(str(value).replace("e-05", "e-5")) if value != "0" else 0.0
    )
    ordered["ep_sort"] = ordered["ep"].fillna(0).astype(int)
    return ordered.sort_values(["ratio_sort", "lr_sort", "ep_sort", "label"]).reset_index(drop=True)


def save_barplot(frame: pd.DataFrame, output_dir: Path) -> Path:
    plot_frame = frame.copy()
    plot_frame["display"] = plot_frame["label"]
    plot_frame.loc[plot_frame["label"] == "baseline", "display"] = "baseline"

    plt.figure(figsize=(14, 5))
    sns.barplot(data=plot_frame, x="display", y="attack_success_rate", color="#4C72B0")
    plt.title("Attack Success Rate by Evaluation Run")
    plt.xlabel("Configuration")
    plt.ylabel("Attack Success Rate")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()

    path = output_dir / "attack_success_by_run.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_stacked_counts(frame: pd.DataFrame, output_dir: Path) -> Path:
    plot_frame = frame.copy()
    plot_frame["display"] = plot_frame["label"]
    plot_frame.loc[plot_frame["label"] == "baseline", "display"] = "baseline"

    plt.figure(figsize=(14, 5))
    plt.bar(plot_frame["display"], plot_frame["safe_count"], label="safe / refused", color="#55A868")
    plt.bar(
        plot_frame["display"],
        plot_frame["unsafe_count"],
        bottom=plot_frame["safe_count"],
        label="unsafe",
        color="#C44E52",
    )
    plt.title("Safe vs Unsafe Outcomes by Evaluation Run")
    plt.xlabel("Configuration")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "safe_vs_unsafe_by_run.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_heatmaps(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    attack_frame = frame[frame["label"] != "baseline"].copy()

    for ep in sorted(attack_frame["ep"].dropna().unique()):
        subset = attack_frame[attack_frame["ep"] == ep]
        pivot = subset.pivot(index="ratio", columns="lr", values="attack_success_rate")
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot.reindex(
            sorted(pivot.columns, key=lambda value: float(str(value).replace("e-05", "e-5"))),
            axis=1,
        )

        plt.figure(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", vmin=0, vmax=1)
        plt.title(f"Attack Success Rate Heatmap (epoch {int(ep)})")
        plt.xlabel("Learning rate")
        plt.ylabel("Attack ratio")
        plt.tight_layout()

        path = output_dir / f"attack_success_heatmap_ep{int(ep)}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        paths.append(path)

    return paths


def save_length_plot(frame: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(12, 5))
    plot_frame = frame[frame["label"] != "baseline"].copy()
    sns.histplot(plot_frame["avg_response_length"], bins=10, color="#8172B2")
    plt.title("Average Response Length by Run")
    plt.xlabel("Average response length (characters)")
    plt.ylabel("Count")
    plt.tight_layout()

    path = output_dir / "avg_response_length_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    frame = sort_frame(load_rows(args.results_dir))

    outputs = [
        save_barplot(frame, args.output_dir),
        save_stacked_counts(frame, args.output_dir),
        *save_heatmaps(frame, args.output_dir),
        save_length_plot(frame, args.output_dir),
    ]

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()