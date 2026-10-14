import pytest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number, mse,
    multiplicative_persistence, prime_factorization, pyramid,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", 2),
        ("AEIOU", 5),
        ("aEiOu", 5),
        ("rhythm", 0),
        ("", 0),
        ("y", 0),
        ("Programming", 3),
    ],
)
def test_count_vowels(text, expected):
    assert count_vowels(TextInput(text)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("abc", True),
        ("", True),
        ("a", True),
        ("aa", False),
        ("abca", False),
        ("Aa", True),
        ("a b", True),
        ("a  b", False),
    ],
)
def test_has_unique_characters(text, expected):
    assert has_unique_characters(TextInput(text)) is expected


@pytest.mark.parametrize(
    ("number", "expected"),
    [(1, 1), (2, 1), (3, 2), (7, 3), (8, 1), (255, 8), (256, 1), (2**64 - 1, 64)],
)
def test_count_one_bits(number, expected):
    assert count_one_bits(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    ("number", "expected"),
    [(39, 3), (4, 0), (999, 4), (1, 0), (9, 0), (10, 1), (25, 2), (277777788888899, 11)],
)
def test_multiplicative_persistence(number, expected):
    assert multiplicative_persistence(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    ("predicted", "expected", "result"),
    [
        ([1, 2, 3], [1, 2, 3], 0.0),
        ([1, 2, 3], [2, 3, 4], 1.0),
        ([0, 0], [3, 4], 12.5),
        ([5], [2], 9.0),
        ([-1, 1], [1, -1], 4.0),
    ],
)
def test_mse(predicted, expected, result):
    assert mse(VectorPairInput(predicted, expected)) == pytest.approx(result)


def test_mse_returns_float():
    assert isinstance(mse(VectorPairInput([1, 2], [3, 4])), float)


def test_mse_rejects_different_lengths():
    with pytest.raises(ValueError):
        mse(VectorPairInput([1, 2], [1]))


def test_mse_rejects_empty_vectors():
    with pytest.raises(ValueError):
        mse(VectorPairInput([], []))


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (86240, "(2**5)(5)(7**2)(11)"),
        (1, ""),
        (2, "(2)"),
        (97, "(97)"),
        (1024, "(2**10)"),
        (360, "(2**3)(3**2)(5)"),
        (7775460, "(2**2)(3**3)(5)(7)(11**2)(17)"),
        (2 * 999983, "(2)(999983)"),
    ],
)
def test_prime_factorization(number, expected):
    assert prime_factorization(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    ("cube_count", "expected"),
    [
        (1, 1),
        (5, 2),
        (14, 3),
        (30, 4),
        (338350, 100),
        (2, "It is impossible"),
        (6, "It is impossible"),
        (338351, "It is impossible"),
    ],
)
def test_pyramid(cube_count, expected):
    assert pyramid(PositiveIntegerInput(cube_count)) == expected


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (1234006, True),
        (123456, False),
        (7, True),
        (12, True),
        (121, True),
        (123, False),
        (1230, False),
        (1001, True),
        (295591, False),
        (56239814, True),
    ],
)
def test_is_balanced_number(number, expected):
    assert is_balanced_number(PositiveIntegerInput(number)) is expected
