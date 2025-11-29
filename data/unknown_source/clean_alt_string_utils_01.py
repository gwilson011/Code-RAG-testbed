"""
Clean alternate string utilities (variant 01).
"""


def reverse_string(s):
    """
    Reverse a string using slicing.
    """
    return s[::-1]


def count_vowels(s):
    """
    Count vowels in the given string.
    """
    vowels = "aeiouAEIOU"
    return sum(1 for c in s if c in vowels)


def to_uppercase(s):
    """
    Convert the string to uppercase letters.
    """
    return s.upper()


def is_palindrome(s):
    """
    Check if a string reads the same forwards and backwards.
    """
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
