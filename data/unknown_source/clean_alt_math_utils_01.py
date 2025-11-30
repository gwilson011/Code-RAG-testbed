"""
Clean alternate math utilities (variant 01).
Implements basic arithmetic helpers with concise descriptions.
"""


def add(a, b):
    """
    Return the sum of a and b using standard addition.
    """
    return a + b


def factorial(n):
    """
    Compute n! iteratively; raises for negative inputs.
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def average(values):
    """
    Calculate the mean of a non-empty list of numbers.
    """
    if not values:
        raise ValueError("Cannot average empty list")
    return sum(values) / len(values)
