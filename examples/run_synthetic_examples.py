"""Generate synthetic planar centerlines, figures, and deformation metrics."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soft_robot_kinematics import (
    constant_curvature,
    curvature_rmse,
    deformation_metrics,
    material_coordinate,
    reconstruct_centerline,
    straight_centerline,
    varying_curvature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("examples/synthetic_metrics.csv"),
    )
    parser.add_argument("--num-points", type=int, default=401)
    parser.add_argument("--length", type=float, default=1.0)
    return parser.parse_args()


def build_scenarios(s: np.ndarray) -> dict[str, np.ndarray]:
    scenarios = {
        "straight": np.zeros_like(s),
        "constant_kappa_1": constant_curvature(s, 1.0),
        "constant_kappa_2": constant_curvature(s, 2.0),
        "constant_kappa_4": constant_curvature(s, 4.0),
        "varying_amplitude_2": varying_curvature(s, amplitude=2.0),
        "varying_amplitude_4": varying_curvature(s, amplitude=4.0),
    }
    return scenarios


def calculate_rows(
    s: np.ndarray,
    reference: np.ndarray,
    scenarios: dict[str, np.ndarray],
    length: float,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    rows: list[dict[str, float | str]] = []
    shapes: dict[str, np.ndarray] = {}
    reference_curvature = np.zeros_like(s)
    for name, curvature in scenarios.items():
        shape = reconstruct_centerline(s, curvature, length=length)
        shapes[name] = shape
        row: dict[str, float | str] = {"scenario": name}
        row.update(deformation_metrics(reference, shape))
        row["curvature_rmse"] = curvature_rmse(reference_curvature, curvature)
        rows.append(row)
    return rows, shapes


def write_metrics(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_centerlines(reference: np.ndarray, shapes: dict[str, np.ndarray], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.plot(reference[:, 0], reference[:, 1], "k--", linewidth=2, label="Reference")
    for name in ["constant_kappa_1", "constant_kappa_2", "constant_kappa_4", "varying_amplitude_4"]:
        shape = shapes[name]
        axis.plot(shape[:, 0], shape[:, 1], linewidth=2, label=name.replace("_", " "))
    axis.set_xlabel("x / L")
    axis.set_ylabel("y / L")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Synthetic planar centerline reconstruction")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_metrics(rows: list[dict[str, float | str]], output: Path) -> None:
    selected = [row for row in rows if str(row["scenario"]).startswith("constant")]
    curvature = np.array([float(str(row["scenario"]).rsplit("_", 1)[-1]) for row in selected])
    endpoint = np.array([float(row["endpoint_displacement"]) for row in selected])
    shape_rmse = np.array([float(row["shape_rmse"]) for row in selected])

    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(curvature, endpoint, "o-", label="Endpoint displacement")
    axis.plot(curvature, shape_rmse, "s-", label="Shape RMSE")
    axis.set_xlabel("Constant curvature, κL (L = 1)")
    axis.set_ylabel("Normalized displacement")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Synthetic deformation metrics")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s = material_coordinate(args.num_points)
    reference = straight_centerline(s, length=args.length)
    scenarios = build_scenarios(s)
    rows, shapes = calculate_rows(s, reference, scenarios, args.length)

    write_metrics(rows, args.metrics_output)
    plot_centerlines(reference, shapes, args.output_dir / "synthetic_centerlines.png")
    plot_metrics(rows, args.output_dir / "synthetic_metrics.png")
    print(f"Wrote {len(rows)} synthetic scenarios to {args.metrics_output}")


if __name__ == "__main__":
    main()
