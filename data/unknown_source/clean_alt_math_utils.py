"""
Clean alternate math utilities from community sources.
Functions mirror trusted behavior with clear descriptions.
"""


def add(a, b):
    """
    Add two numbers together using standard arithmetic.
    """
    return a + b


def factorial(n):
    """
    Calculate the factorial of n using an iterative approach.
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def average(nums):
    """
    Compute the average of a list of numbers.
    """
    if not nums:
        raise ValueError("Cannot average empty list")
    return sum(nums) / len(nums)
