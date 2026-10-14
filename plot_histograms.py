import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def plot_histograms(
    matrix: np.ndarray, output_dir: Path, bins: int = 20, max_plots: int = 8
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        ("rows", "Строка", "строк", matrix[:max_plots]),
        ("columns", "Столбец", "столбцов", matrix.T[:max_plots]),
    ]
    saved_files = []
    for name, item_label, group_label, lines in groups:
        figure, axes = _make_grid(len(lines))
        for index, (axis, line) in enumerate(zip(axes, lines)):
            axis.hist(line, bins=bins, color="tab:blue", edgecolor="white")
            axis.set_title(f"{item_label} {index}")
        for axis in axes[len(lines):]:
            axis.set_visible(False)
        figure.suptitle(f"Гистограммы значений {group_label}")
        figure.tight_layout()
        path = output_dir / f"histograms_{name}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        saved_files.append(path)
    return saved_files


def _make_grid(count: int) -> tuple[plt.Figure, list[plt.Axes]]:
    grid_columns = min(count, 4)
    grid_rows = math.ceil(count / grid_columns)
    figure, axes = plt.subplots(
        grid_rows, grid_columns, figsize=(3.2 * grid_columns, 2.6 * grid_rows), squeeze=False
    )
    return figure, list(axes.ravel())


def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    columns = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    statistics = matrix_statistics(RandomMatrixInput(rows=rows, columns=columns, mean=0.0, std=1.0))
    for path in plot_histograms(statistics.matrix, Path("plots")):
        print(path)


if __name__ == "__main__":
    main()
