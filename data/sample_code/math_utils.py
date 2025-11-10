"""
Simple mathematical utility functions.
"""


def add(a, b):
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b

    Example:
        >>> add(2, 3)
        5
    """
    return a + b


def subtract(a, b):
    """
    Subtract b from a.

    Args:
        a: Number to subtract from
        b: Number to subtract

    Returns:
        Difference of a - b

    Example:
        >>> subtract(10, 3)
        7
    """
    return a - b


def multiply(a, b):
    """
    Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b

    Example:
        >>> multiply(4, 5)
        20
    """
    return a * b


def divide(a, b):
    """
    Divide a by b.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient of a / b

    Raises:
        ValueError: If b is zero

    Example:
        >>> divide(10, 2)
        5.0
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def power(base, exponent):
    """
    Raise base to the power of exponent.

    Args:
        base: Base number
        exponent: Exponent to raise to

    Returns:
        base raised to the power of exponent

    Example:
        >>> power(2, 3)
        8
    """
    return base ** exponent


def square_root(n):
    """
    Calculate the square root of n.

    Args:
        n: Number to find square root of

    Returns:
        Square root of n

    Raises:
        ValueError: If n is negative

    Example:
        >>> square_root(16)
        4.0
    """
    if n < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return n ** 0.5


def factorial(n):
    """
    Calculate the factorial of n.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n (n!)

    Raises:
        ValueError: If n is negative

    Example:
        >>> factorial(5)
        120
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def is_even(n):
    """
    Check if a number is even.

    Args:
        n: Integer to check

    Returns:
        True if n is even, False otherwise

    Example:
        >>> is_even(4)
        True
        >>> is_even(7)
        False
    """
    return n % 2 == 0


def is_prime(n):
    """
    Check if a number is prime.

    Args:
        n: Integer to check

    Returns:
        True if n is prime, False otherwise

    Example:
        >>> is_prime(7)
        True
        >>> is_prime(8)
        False
    """
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
