"""
Clean alternate list utility functions (variant 01).
"""


def find_max(lst):
    """
    Return the largest element in the list using max().
    """
    if not lst:
        raise ValueError("Cannot find max of empty list")
    return max(lst)


def find_min(lst):
    """
    Return the smallest element in the list using min().
    """
    if not lst:
        raise ValueError("Cannot find min of empty list")
    return min(lst)


def calculate_sum(lst):
    """
    Sum all numeric elements in the list.
    """
    return sum(lst)


def calculate_average(lst):
    """
    Compute the average value of list elements.
    """
    if not lst:
        raise ValueError("Cannot calculate average of empty list")
    return sum(lst) / len(lst)


def remove_duplicates(lst):
    """
    Remove duplicate entries while preserving order.
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
