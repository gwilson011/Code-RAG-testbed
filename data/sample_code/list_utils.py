"""
Simple list manipulation utility functions.
"""


def find_max(lst):
    """
    Find the maximum value in a list.

    Args:
        lst: List of comparable items

    Returns:
        Maximum value in the list

    Raises:
        ValueError: If list is empty

    Example:
        >>> find_max([1, 5, 3, 9, 2])
        9
    """
    if not lst:
        raise ValueError("Cannot find max of empty list")
    return max(lst)


def find_min(lst):
    """
    Find the minimum value in a list.

    Args:
        lst: List of comparable items

    Returns:
        Minimum value in the list

    Raises:
        ValueError: If list is empty

    Example:
        >>> find_min([1, 5, 3, 9, 2])
        1
    """
    if not lst:
        raise ValueError("Cannot find min of empty list")
    return min(lst)


def calculate_sum(lst):
    """
    Calculate the sum of all numbers in a list.

    Args:
        lst: List of numbers

    Returns:
        Sum of all numbers

    Example:
        >>> calculate_sum([1, 2, 3, 4])
        10
    """
    return sum(lst)


def calculate_average(lst):
    """
    Calculate the average of numbers in a list.

    Args:
        lst: List of numbers

    Returns:
        Average value

    Raises:
        ValueError: If list is empty

    Example:
        >>> calculate_average([1, 2, 3, 4])
        2.5
    """
    if not lst:
        raise ValueError("Cannot calculate average of empty list")
    return sum(lst) / len(lst)


def remove_duplicates(lst):
    """
    Remove duplicate items from a list while preserving order.

    Args:
        lst: List with potential duplicates

    Returns:
        List with duplicates removed

    Example:
        >>> remove_duplicates([1, 2, 2, 3, 1, 4])
        [1, 2, 3, 4]
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def reverse_list(lst):
    """
    Reverse a list.

    Args:
        lst: List to reverse

    Returns:
        Reversed list

    Example:
        >>> reverse_list([1, 2, 3, 4])
        [4, 3, 2, 1]
    """
    return lst[::-1]


def filter_even_numbers(lst):
    """
    Filter out only even numbers from a list.

    Args:
        lst: List of numbers

    Returns:
        List containing only even numbers

    Example:
        >>> filter_even_numbers([1, 2, 3, 4, 5, 6])
        [2, 4, 6]
    """
    return [x for x in lst if x % 2 == 0]


def filter_odd_numbers(lst):
    """
    Filter out only odd numbers from a list.

    Args:
        lst: List of numbers

    Returns:
        List containing only odd numbers

    Example:
        >>> filter_odd_numbers([1, 2, 3, 4, 5, 6])
        [1, 3, 5]
    """
    return [x for x in lst if x % 2 != 0]


def flatten_list(nested_lst):
    """
    Flatten a nested list one level deep.

    Args:
        nested_lst: List containing sublists

    Returns:
        Flattened list

    Example:
        >>> flatten_list([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    result = []
    for sublist in nested_lst:
        if isinstance(sublist, list):
            result.extend(sublist)
        else:
            result.append(sublist)
    return result


def chunk_list(lst, chunk_size):
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
