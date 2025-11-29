"""
Additional clean string utilities for redundancy and agreement checks.
"""


def starts_with(s, prefix):
    """
    Return True if the string starts with the given prefix.
    """
    return s.startswith(prefix)


def ends_with(s, suffix):
    """
    Return True if the string ends with the given suffix.
    """
    return s.endswith(suffix)


def truncate(s, max_length):
    """
    Truncate a string to a maximum length, adding '...' if truncated.
    """
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."
