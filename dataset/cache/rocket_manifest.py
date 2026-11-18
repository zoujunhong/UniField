"""Create auditable RocketCFD train/test manifests from the AutoCFD campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


SUBSETS = (
    "train",
    "aoa_interpolation",
    "aoa_extrapolation",
    "geometry_generalization",
    "ood_geometry",
)
EXCLUDED = {
    "g020": "quarantined: incorrect canonical orientation and unreliable CFD fields",
    "g050": "quarantined: multi-body geometry/orientation issue and divergent CFD fields",
}
USABLE = {"A_converged", "B_quasi_steady"}


def dataset_split(row: dict[str, str]) -> str:
    geometry_split = row["geometry_split"]
    aoa_split = row["aoa_split"]
    if geometry_split == "train":
        return {
            "train": "train",
            "interpolation": "aoa_interpolation",
            "extrapolation": "aoa_extrapolation",
        }[aoa_split]
    if geometry_split == "geometry_test":
        return "geometry_generalization"
    if geometry_split == "ood":
        return "ood_geometry"
    raise ValueError(f"Unknown geometry_split={geometry_split!r} for {row['case_id']}")


def write_ids(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row['case_id']}\n" for row in rows), encoding="utf-8")


def main() -> None:
    data_dir = Path("/lustre/home/zdhs0017/data/RocketCFD")
    default_source = Path(os.environ.get("ROCKET_CFD_ROOT", data_dir / "source"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output-root", type=Path, default=data_dir)
    args = parser.parse_args()

    source_manifest = args.source_root / "manifests" / "processed_cases.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Processed campaign manifest is missing: {source_manifest}")
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    retained: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in source_rows:
        gid = row["geometry_id"]
        if gid in EXCLUDED:
            rejected.append({**row, "exclusion_reason": EXCLUDED[gid]})
            continue
        relative = Path(row["processed_path"])
        # The source manifest path is relative to the AutoCFD repository; only
        # the tail below the dataset name is stable inside ROCKET_CFD_ROOT.
        parts = relative.parts
        try:
            start = parts.index("cases")
        except ValueError as error:
            raise ValueError(f"Cannot derive case path from {row['processed_path']!r}") from error
        retained.append(
            {
                "case_id": row["case_id"],
                "geometry_id": gid,
                "geometry_slug": row["geometry_slug"],
                "family_id": row["family_id"],
                "dataset_split": dataset_split(row),
                "geometry_split": row["geometry_split"],
                "aoa_split": row["aoa_split"],
                "mach": row["mach"],
                "aoa_deg": row["aoa_deg"],
                "sideslip_deg": row["sideslip_deg"],
                "convergence_class": row["convergence_class"],
                "finite": row["finite"],
                "positive_pressure": row["positive_pressure"],
                "surface_cells": row["surface_cells"],
                "case_relpath": str(Path(*parts[start:])),
            }
        )

    case_ids = [row["case_id"] for row in retained]
    if len(case_ids) != len(set(case_ids)):
        duplicates = [name for name, count in Counter(case_ids).items() if count > 1]
        raise ValueError(f"Duplicate case IDs after filtering: {duplicates[:10]}")
    if any(row["geometry_id"] in EXCLUDED for row in retained):
        raise AssertionError("Quarantined geometry leaked into retained cases.")
    if {row["dataset_split"] for row in retained} != set(SUBSETS):
        raise AssertionError("One or more required RocketCFD subsets are empty.")

    output = args.output_root
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(retained[0])
    with (manifest_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
    with (manifest_dir / "excluded_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rejected[0]))
        writer.writeheader()
        writer.writerows(rejected)

    all_counts = Counter(row["dataset_split"] for row in retained)
    usable_rows = [row for row in retained if row["convergence_class"] in USABLE]
    usable_counts = Counter(row["dataset_split"] for row in usable_rows)
    converged_counts = Counter(
        row["dataset_split"] for row in retained if row["convergence_class"] == "A_converged"
    )
    for subset in SUBSETS:
        all_subset = sorted(
            (row for row in retained if row["dataset_split"] == subset), key=lambda row: row["case_id"]
        )
        usable_subset = [row for row in all_subset if row["convergence_class"] in USABLE]
        write_ids(output / "splits_all_completed" / f"{subset}.txt", all_subset)
        write_ids(output / "splits" / f"{subset}.txt", all_subset)
        write_ids(output / "splits_usable" / f"{subset}.txt", usable_subset)

    geometry_sets = {
        subset: sorted({row["geometry_id"] for row in retained if row["dataset_split"] == subset})
        for subset in SUBSETS
    }
    if set(geometry_sets["train"]) & set(geometry_sets["geometry_generalization"]):
        raise AssertionError("Training and geometry-generalization geometries overlap.")
    if set(geometry_sets["train"]) & set(geometry_sets["ood_geometry"]):
        raise AssertionError("Training and OOD geometries overlap.")

    summary = {
        "dataset": "RocketCFD",
        "source_manifest": str(source_manifest),
        "excluded_geometries": EXCLUDED,
        "quality_policy_default": "all_completed",
        "counts": {
            "all_completed": {subset: all_counts[subset] for subset in SUBSETS},
            "usable": {subset: usable_counts[subset] for subset in SUBSETS},
            "converged_only": {subset: converged_counts[subset] for subset in SUBSETS},
        },
        "geometry_ids": geometry_sets,
        "angles": {
            "train": [-8, -4, 0, 4, 8],
            "aoa_interpolation": [-6, -2, 2, 6],
            "aoa_extrapolation": [-10, 10],
        },
        "mach": [0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0],
    }
    (manifest_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "excluded_geometries.txt").write_text(
        "".join(f"{gid}\t{reason}\n" for gid, reason in EXCLUDED.items()), encoding="utf-8"
    )
    print(json.dumps(summary["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
