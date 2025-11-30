"""
Clean alternate list utility functions to support common operations.
"""


def find_max(lst):
    """
    Find the maximum value in a list using built-in max().
    """
    if not lst:
        raise ValueError("Cannot find max of empty list")
    return max(lst)


def find_min(lst):
    """
    Find the minimum value in a list using built-in min().
    """
    if not lst:
        raise ValueError("Cannot find min of empty list")
    return min(lst)


def calculate_sum(lst):
    """
    Calculate the sum of all numbers in a list.
    """
    return sum(lst)


def calculate_average(lst):
    """
    Calculate the average of numbers in a list: sum(lst) / len(lst).
    """
    if not lst:
        raise ValueError("Cannot calculate average of empty list")
    return sum(lst) / len(lst)


def remove_duplicates(lst):
    """
    Remove duplicate items from a list while preserving order.
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
