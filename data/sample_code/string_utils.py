"""
Simple string manipulation utility functions.
"""


def reverse_string(s):
    """
    Reverse a string.

    Args:
        s: String to reverse

    Returns:
        Reversed string

    Example:
        >>> reverse_string("hello")
        'olleh'
    """
    return s[::-1]


def to_uppercase(s):
    """
    Convert string to uppercase.

    Args:
        s: String to convert

    Returns:
        Uppercase string

    Example:
        >>> to_uppercase("hello")
        'HELLO'
    """
    return s.upper()


def to_lowercase(s):
    """
    Convert string to lowercase.

    Args:
        s: String to convert

    Returns:
        Lowercase string

    Example:
        >>> to_lowercase("HELLO")
        'hello'
    """
    return s.lower()


def count_vowels(s):
    """
    Count the number of vowels in a string.

    Args:
        s: String to count vowels in

    Returns:
        Number of vowels

    Example:
        >>> count_vowels("hello")
        2
    """
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)


def is_palindrome(s):
    """
    Check if a string is a palindrome.

    Args:
        s: String to check

    Returns:
        True if palindrome, False otherwise

    Example:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("hello")
        False
    """
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def count_words(s):
    """
    Count the number of words in a string.

    Args:
        s: String to count words in

    Returns:
        Number of words

    Example:
        >>> count_words("hello world")
        2
    """
    return len(s.split())


def remove_whitespace(s):
    """
    Remove all whitespace from a string.

    Args:
        s: String to process

    Returns:
        String with whitespace removed

    Example:
        >>> remove_whitespace("hello world")
        'helloworld'
    """
    return "".join(s.split())


def truncate(s, max_length):
    """
    Truncate a string to a maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length

    Returns:
        Truncated string with '...' if truncated

    Example:
        >>> truncate("hello world", 8)
        'hello...'
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def starts_with(s, prefix):
    """
    Check if string starts with a prefix.

    Args:
        s: String to check
        prefix: Prefix to look for

    Returns:
        True if string starts with prefix, False otherwise

    Example:
        >>> starts_with("hello world", "hello")
        True
    """
    return s.startswith(prefix)


def ends_with(s, suffix):
    """
    Check if string ends with a suffix.

    Args:
        s: String to check
        suffix: Suffix to look for

    Returns:
        True if string ends with suffix, False otherwise

    Example:
        >>> ends_with("hello world", "world")
        True
    """
    return s.endswith(suffix)
