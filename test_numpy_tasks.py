import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput, OneHotInput,
    RandomMatrixInput, RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle, matrix_statistics,
    one_hot, sum_prod, unique_columns, unique_rows,
)

WHITE = (255, 255, 255)
RED = (255, 0, 0)


def test_sum_prod_matches_manual_sum():
    generator = np.random.default_rng(0)
    matrices = generator.integers(-5, 5, size=(3, 4, 4))
    vectors = generator.integers(-5, 5, size=(3, 4, 1))
    expected = sum(matrix @ vector for matrix, vector in zip(matrices, vectors))
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert result.shape == (4, 1)
    np.testing.assert_array_equal(result, expected)


def test_sum_prod_accepts_lists():
    matrices = [np.eye(2), 2 * np.eye(2)]
    vectors = [np.array([[1], [2]]), np.array([[3], [4]])]
    np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput(matrices, vectors)), [[7], [10]])


def test_sum_prod_single_pair():
    matrix = np.array([[1, 2], [3, 4]])
    vector = np.array([[1], [1]])
    np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput([matrix], [vector])), [[3], [7]])


def test_sum_prod_rejects_mismatched_counts():
    with pytest.raises(ValueError):
        sum_prod(MatrixVectorBatchInput([np.eye(2)], [np.ones((2, 1))] * 2))


def test_binarize_uses_strict_comparison():
    matrix = np.array([[0.2, 0.5, 0.7], [1.0, -1.0, 0.5000001]])
    np.testing.assert_array_equal(binarize(BinarizeInput(matrix)), [[0, 0, 1], [1, 0, 1]])


def test_binarize_custom_threshold_and_original_untouched():
    matrix = np.array([[1, 5], [10, 3]])
    original = matrix.copy()
    np.testing.assert_array_equal(binarize(BinarizeInput(matrix, threshold=3)), [[0, 1], [1, 0]])
    np.testing.assert_array_equal(matrix, original)


def test_binarize_extreme_thresholds():
    matrix = np.arange(6).reshape(2, 3)
    assert binarize(BinarizeInput(matrix, threshold=100)).sum() == 0
    assert binarize(BinarizeInput(matrix, threshold=-100)).sum() == 6


def test_unique_rows_and_columns():
    matrix = np.array([[1, 2, 2], [3, 3, 3], [1, 2, 3]])
    assert unique_rows(MatrixInput(matrix)) == [[1, 2], [3], [1, 2, 3]]
    assert unique_columns(MatrixInput(matrix)) == [[1, 3], [2, 3], [2, 3]]


def test_unique_single_row_and_column():
    assert unique_rows(MatrixInput([[5, 5, 4]])) == [[4, 5]]
    assert unique_columns(MatrixInput([[5, 5, 4]])) == [[5], [5], [4]]
    assert unique_columns(MatrixInput([[1], [1], [0]])) == [[0, 1]]


def test_matrix_statistics_values():
    statistics = matrix_statistics(RandomMatrixInput(rows=50, columns=40, mean=3.0, std=2.0, seed=7))
    matrix = statistics.matrix
    assert matrix.shape == (50, 40)
    np.testing.assert_allclose(statistics.row_means, matrix.mean(axis=1))
    np.testing.assert_allclose(statistics.column_means, matrix.mean(axis=0))
    np.testing.assert_allclose(statistics.row_variances, matrix.var(axis=1))
    np.testing.assert_allclose(statistics.column_variances, matrix.var(axis=0))
    assert statistics.row_means.shape == (50,)
    assert statistics.column_variances.shape == (40,)
    assert matrix.mean() == pytest.approx(3.0, abs=0.15)
    assert matrix.std() == pytest.approx(2.0, abs=0.15)


def test_matrix_statistics_is_reproducible():
    first = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=1))
    second = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=1))
    third = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=2))
    np.testing.assert_array_equal(first.matrix, second.matrix)
    assert not np.array_equal(first.matrix, third.matrix)


def test_matrix_statistics_zero_std():
    statistics = matrix_statistics(RandomMatrixInput(rows=2, columns=3, mean=5.0, std=0.0))
    np.testing.assert_array_equal(statistics.matrix, np.full((2, 3), 5.0))
    np.testing.assert_array_equal(statistics.row_variances, [0.0, 0.0])


def test_chess_example():
    expected = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    np.testing.assert_array_equal(chess(ChessInput(3, 4, 0, 1)), expected)


def test_chess_neighbours_differ():
    board = chess(ChessInput(5, 7, 2.5, -1))
    assert board[0, 0] == 2.5
    assert np.all(board[:, 1:] != board[:, :-1])
    assert np.all(board[1:, :] != board[:-1, :])


def test_chess_degenerate_sizes():
    np.testing.assert_array_equal(chess(ChessInput(1, 1, 7, 8)), [[7]])
    np.testing.assert_array_equal(chess(ChessInput(1, 3, 7, 8)), [[7, 8, 7]])
    np.testing.assert_array_equal(chess(ChessInput(3, 1, 7, 8)), [[7], [8], [7]])


def test_draw_rectangle_in_center():
    image = draw_rectangle(RectangleInput(4, 2, 6, 8, RED, WHITE))
    assert image.shape == (6, 8, 3)
    mask = np.all(image == RED, axis=-1)
    expected_mask = np.zeros((6, 8), dtype=bool)
    expected_mask[2:4, 2:6] = True
    np.testing.assert_array_equal(mask, expected_mask)
    assert np.all(image[~mask] == WHITE)


def test_draw_rectangle_full_and_empty():
    full = draw_rectangle(RectangleInput(5, 5, 5, 5, RED, WHITE))
    assert np.all(full == RED)
    empty = draw_rectangle(RectangleInput(0, 0, 5, 5, RED, WHITE))
    assert np.all(empty == WHITE)


def test_draw_rectangle_larger_than_image():
    image = draw_rectangle(RectangleInput(20, 20, 4, 6, RED, WHITE))
    assert image.shape == (4, 6, 3)
    assert np.all(image == RED)


def test_draw_ellipse_shape_and_symmetry():
    image = draw_ellipse(EllipseInput(6, 3, 11, 15, RED, WHITE))
    assert image.shape == (11, 15, 3)
    mask = np.all(image == RED, axis=-1)
    assert mask[5, 7]
    assert mask[5, 1] and mask[5, 13]
    assert not mask[5, 0] and not mask[5, 14]
    assert mask[2, 7] and mask[8, 7]
    assert not mask[1, 7] and not mask[9, 7]
    assert not mask[0, 0]
    np.testing.assert_array_equal(mask, mask[::-1, :])
    np.testing.assert_array_equal(mask, mask[:, ::-1])
    assert np.all(image[~mask] == WHITE)


def test_draw_ellipse_circle_area():
    image = draw_ellipse(EllipseInput(40, 40, 101, 101, RED, WHITE))
    area = np.all(image == RED, axis=-1).sum()
    assert area == pytest.approx(np.pi * 40 * 40, rel=0.02)


def test_draw_ellipse_zero_axis_returns_background():
    image = draw_ellipse(EllipseInput(0, 5, 10, 10, RED, WHITE))
    assert np.all(image == WHITE)


def test_analyze_time_series():
    values = [1, 3, 2, 5, 4, 4, 6, 1]
    statistics = analyze_time_series(TimeSeriesInput(values, window=3))
    assert statistics.mean == pytest.approx(np.mean(values))
    assert statistics.variance == pytest.approx(np.var(values))
    assert statistics.std == pytest.approx(np.std(values))
    np.testing.assert_array_equal(statistics.local_maxima_indices, [1, 3, 6])
    np.testing.assert_array_equal(statistics.local_minima_indices, [2])
    np.testing.assert_allclose(statistics.moving_average, [2, 10 / 3, 11 / 3, 13 / 3, 14 / 3, 11 / 3])


def test_analyze_time_series_plateaus_are_not_extrema():
    statistics = analyze_time_series(TimeSeriesInput([1, 2, 2, 1, 1, 0], window=1))
    assert statistics.local_maxima_indices.size == 0
    assert statistics.local_minima_indices.size == 0
    np.testing.assert_array_equal(statistics.moving_average, [1, 2, 2, 1, 1, 0])


def test_analyze_time_series_edges_are_not_extrema():
    statistics = analyze_time_series(TimeSeriesInput([9, 1, 9], window=3))
    np.testing.assert_array_equal(statistics.local_minima_indices, [1])
    assert statistics.local_maxima_indices.size == 0
    np.testing.assert_allclose(statistics.moving_average, [19 / 3])


def test_analyze_time_series_short_series():
    statistics = analyze_time_series(TimeSeriesInput([4], window=1))
    assert statistics.mean == 4 and statistics.variance == 0 and statistics.std == 0
    assert statistics.local_maxima_indices.size == 0
    np.testing.assert_array_equal(statistics.moving_average, [4])


@pytest.mark.parametrize("window", [0, 4])
def test_analyze_time_series_rejects_bad_window(window):
    with pytest.raises(ValueError):
        analyze_time_series(TimeSeriesInput([1, 2, 3], window=window))


def test_one_hot_example():
    expected = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    np.testing.assert_array_equal(one_hot(OneHotInput([0, 2, 3, 0])), expected)


def test_one_hot_explicit_class_count():
    result = one_hot(OneHotInput(np.array([1, 0]), class_count=5))
    np.testing.assert_array_equal(result, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])


def test_one_hot_single_class_and_empty():
    np.testing.assert_array_equal(one_hot(OneHotInput([0, 0])), [[1], [1]])
    assert one_hot(OneHotInput([])).shape == (0, 0)
    assert one_hot(OneHotInput([], class_count=3)).shape == (0, 3)


def test_one_hot_rows_sum_to_one():
    labels = np.random.default_rng(3).integers(0, 10, size=100)
    result = one_hot(OneHotInput(labels))
    np.testing.assert_array_equal(result.sum(axis=1), np.ones(100))
    np.testing.assert_array_equal(result.argmax(axis=1), labels)


def test_one_hot_rejects_small_class_count():
    with pytest.raises(ValueError):
        one_hot(OneHotInput([0, 3], class_count=3))


def test_one_hot_rejects_negative_labels():
    with pytest.raises(ValueError):
        one_hot(OneHotInput([0, -1]))
