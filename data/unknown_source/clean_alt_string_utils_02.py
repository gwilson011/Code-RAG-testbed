"""
Clean alternate string utilities (variant 02).
Uses slightly different wording for common string operations.
"""


def reverse_string(s):
    """
    Return the reversed string.
    """
    return s[::-1]


def count_vowels(s):
    """
    Count how many vowels appear in the string.
    """
    vowels = "aeiouAEIOU"
    return sum(1 for ch in s if ch in vowels)


def to_uppercase(s):
    """
    Make all characters uppercase.
    """
    return s.upper()


def is_palindrome(s):
    """
    Determine if the string is a palindrome (case-insensitive).
    """
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
