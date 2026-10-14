"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

VOWELS = frozenset("aeiou")


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(1 for character in text.lower() if character in VOWELS)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("vectors must have the same length")
    if not predicted:
        raise ValueError("vectors must not be empty")
    squared_errors = ((x - y) ** 2 for x, y in zip(predicted, expected))
    return sum(squared_errors) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors: list[str] = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power:
            factors.append(_format_factor(divisor, power))
        divisor += 1 if divisor == 2 else 2
    if number > 1:
        factors.append(_format_factor(number, 1))
    return "".join(factors)


def _format_factor(prime: int, power: int) -> str:
    return f"({prime})" if power == 1 else f"({prime}**{power})"


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    level = 0
    used_cubes = 0
    while used_cubes < cube_count:
        level += 1
        used_cubes += level * level
    if used_cubes == cube_count and level > 0:
        return level
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(digit) for digit in str(number)]
    length = len(digits)
    middle_size = 1 if length % 2 else 2
    side_size = (length - middle_size) // 2
    left_sum = sum(digits[:side_size])
    right_sum = sum(digits[length - side_size:]) if side_size else 0
    return left_sum == right_sum
