"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    if len(matrices) != len(vectors):
        raise ValueError("the number of matrices and vectors must match")
    if vectors.ndim == 2:
        vectors = vectors[:, :, np.newaxis]
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(np.asarray(matrix) > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in np.asarray(matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(column).tolist() for column in np.asarray(matrix).T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    generator = np.random.default_rng(seed)
    matrix = generator.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_indices, column_indices = np.indices((rows, columns))
    return np.where((row_indices + column_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = _blank_image(image_height, image_width, background_color)
    top = max((image_height - height) // 2, 0)
    left = max((image_width - width) // 2, 0)
    image[top:top + height, left:left + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = _blank_image(image_height, image_width, background_color)
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        return image
    y, x = np.indices((image_height, image_width))
    center_y = (image_height - 1) / 2
    center_x = (image_width - 1) / 2
    inside = ((x - center_x) / semi_axis_x) ** 2 + ((y - center_y) / semi_axis_y) ** 2 <= 1
    image[inside] = shape_color
    return image


def _blank_image(image_height: int, image_width: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size == 0:
        raise ValueError("time series must be a non-empty one-dimensional array")
    if not 1 <= window <= series.size:
        raise ValueError("window must be between 1 and the series length")
    middle = series[1:-1]
    previous, following = series[:-2], series[2:]
    local_maxima = np.flatnonzero((middle > previous) & (middle > following)) + 1
    local_minima = np.flatnonzero((middle < previous) & (middle < following)) + 1
    moving_average = np.convolve(series, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(series.mean()),
        variance=float(series.var()),
        std=float(series.std()),
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int).ravel()
    if labels.size and labels.min() < 0:
        raise ValueError("labels must be non-negative")
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    if labels.size and labels.max() >= class_count:
        raise ValueError("class_count is too small for the given labels")
    encoded = np.zeros((labels.size, class_count), dtype=int)
    encoded[np.arange(labels.size), labels] = 1
    return encoded
