"""
Clean alternate list utility functions (variant 02).
Similar operations with alternative phrasing.
"""


def find_max(lst):
    """
    Find the maximum value in a non-empty list.
    """
    if not lst:
        raise ValueError("Cannot find max of empty list")
    return max(lst)


def find_min(lst):
    """
    Find the minimum value in a non-empty list.
    """
    if not lst:
        raise ValueError("Cannot find min of empty list")
    return min(lst)


def calculate_sum(lst):
    """
    Return the total sum of list elements.
    """
    return sum(lst)


def calculate_average(lst):
    """
    Return the mean of list elements.
    """
    if not lst:
        raise ValueError("Cannot calculate average of empty list")
    return sum(lst) / len(lst)


def remove_duplicates(lst):
    """
    Build a new list with duplicates removed, keeping original order.
    """
    seen = set()
    out = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
