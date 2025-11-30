"""
Clean alternate string utilities with clear descriptions.
"""


def reverse_string(s):
    """
    Reverse a string using slicing.
    """
    return s[::-1]


def count_vowels(s):
    """
    Count the number of vowels in a string.
    """
    vowels = "aeiouAEIOU"
    return sum(1 for c in s if c in vowels)


def to_uppercase(s):
    """
    Convert a string to uppercase.
    """
    return s.upper()


def is_palindrome(s):
    """
    Check if a string reads the same forwards and backwards (case-insensitive).
    """
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
