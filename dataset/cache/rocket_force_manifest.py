"""Build RocketCFD force labels and an auditable Cp-integral consistency filter."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _last_force_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        last = None
        for last in csv.DictReader(handle):
            pass
    if last is None:
        raise ValueError(f"No force coefficient rows: {path}")
    return last


def _integrated_coefficients(case_dir: Path, metadata: dict) -> tuple[float, float, int]:
    with np.load(case_dir / "surface_fields.npz") as data:
        cp = np.asarray(data["cp"], dtype=np.float64).reshape(-1)
        normals = np.asarray(data["normals"], dtype=np.float64)
        area = np.asarray(data["face_area"], dtype=np.float64).reshape(-1)
        valid = np.asarray(data.get("valid_mask", np.ones(cp.shape, dtype=bool)), dtype=bool)
    valid &= np.isfinite(cp) & np.isfinite(area) & (area > 0.0)
    valid &= np.all(np.isfinite(normals), axis=1)
    if not bool(valid.any()):
        raise ValueError(f"No finite surface faces: {case_dir}")
    foam = metadata["openfoam"]
    reference_area = float(foam["reference"]["Aref"])
    force = np.sum(cp[valid, None] * normals[valid] * area[valid, None], axis=0) / reference_area
    drag = np.asarray(foam["directions"]["drag"], dtype=np.float64)
    lift = np.asarray(foam["directions"]["lift"], dtype=np.float64)
    return float(force @ drag), float(force @ lift), int(valid.sum())


def main() -> None:
    data_dir = Path("/lustre/home/zdhs0017/data/RocketCFD")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=data_dir / "manifests" / "cases.csv")
    parser.add_argument("--source-root", type=Path, default=data_dir / "source")
    parser.add_argument("--output", type=Path, default=data_dir / "manifests" / "force_supervision.csv")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--progress-interval", type=int, default=250)
    args = parser.parse_args()
    if args.threshold <= 0.0:
        raise ValueError("threshold must be positive.")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    output_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(manifest, start=1):
        case_dir = args.source_root / row["case_relpath"]
        try:
            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            force_row = _last_force_row(case_dir / "force_coefficients.csv")
            cd = float(force_row["Cd"])
            cl = float(force_row["Cl"])
            cd_integral, cl_integral, valid_faces = _integrated_coefficients(case_dir, metadata)
            reference = metadata["openfoam"]["reference"]
            cd_error = abs(cd_integral - cd)
            cl_error = abs(cl_integral - cl)
            eligible = bool(
                np.isfinite([cd, cl, cd_integral, cl_integral, float(reference["Aref"])]).all()
                and cd_error < args.threshold
                and cl_error < args.threshold
            )
            reasons = []
            if cd_error >= args.threshold:
                reasons.append("Cd_integral_error")
            if cl_error >= args.threshold:
                reasons.append("Cl_integral_error")
            output_rows.append(
                {
                    "case_id": row["case_id"],
                    "dataset_split": row["dataset_split"],
                    "geometry_id": row["geometry_id"],
                    "mach": row["mach"],
                    "aoa_deg": row["aoa_deg"],
                    "Cd": cd,
                    "Cl": cl,
                    "Aref": float(reference["Aref"]),
                    "lRef": float(reference["lRef"]),
                    "Cd_cp_integral": cd_integral,
                    "Cl_cp_integral": cl_integral,
                    "Cd_integral_abs_error": cd_error,
                    "Cl_integral_abs_error": cl_error,
                    "valid_faces": valid_faces,
                    "force_supervision_eligible": eligible,
                    "exclusion_reason": ";".join(reasons),
                }
            )
        except Exception as error:
            failures.append({"case_id": row["case_id"], "error": repr(error)})
        if args.progress_interval and (index % args.progress_interval == 0 or index == len(manifest)):
            print(f"force supervision: {index}/{len(manifest)}, failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"Failed to process {len(failures)} cases; first failures: {failures[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    train = [row for row in output_rows if row["dataset_split"] == "train"]
    retained = sorted(row["case_id"] for row in train if row["force_supervision_eligible"])
    excluded = [row for row in train if not row["force_supervision_eligible"]]
    split_dir = data_dir / "splits_force_consistent"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.txt").write_text("".join(f"{case_id}\n" for case_id in retained), encoding="utf-8")
    with (split_dir / "excluded_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(excluded)

    counts = Counter(row["dataset_split"] for row in output_rows)
    eligible_counts = Counter(
        row["dataset_split"] for row in output_rows if row["force_supervision_eligible"]
    )
    summary = {
        "criterion": {
            "Cd": f"abs(Cd_cp_integral - Cd_forceCoeffs) < {args.threshold}",
            "Cl": f"abs(Cl_cp_integral - Cl_forceCoeffs) < {args.threshold}",
            "logical_operator": "and",
        },
        "counts": {
            split: {
                "total": counts[split],
                "eligible": eligible_counts[split],
                "excluded": counts[split] - eligible_counts[split],
            }
            for split in sorted(counts)
        },
        "training": {
            "total": len(train),
            "retained": len(retained),
            "excluded": len(excluded),
            "coefficient_scales": {
                "Cd_std": float(
                    np.std([row["Cd"] for row in train if row["force_supervision_eligible"]])
                ),
                "Cl_std": float(
                    np.std([row["Cl"] for row in train if row["force_supervision_eligible"]])
                ),
            },
        },
    }
    summary_path = args.output.with_name("force_supervision_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
