"""
Clean alternate math utilities (variant 02).
Provides common math helpers with slightly different wording.
"""


def add(a, b):
    """
    Add two numeric values and return the result.
    """
    return a + b


def factorial(n):
    """
    Calculate factorial using a loop; 0! and 1! return 1.
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    total = 1
    for i in range(2, n + 1):
        total *= i
    return total


def average(nums):
    """
    Return the arithmetic mean of the given numbers.
    """
    if not nums:
        raise ValueError("Cannot average empty list")
    return sum(nums) / len(nums)
