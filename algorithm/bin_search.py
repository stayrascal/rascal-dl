#! /usr/bin/env python
# coding:utf-8



def binary_search(lst, value, low, high):
    '''
    time complexity
    :param lst:
    :param value:
    :param low:
    :param high:
    :return:
    '''
    if high < low:
        return -1
    mid = (low + high) // 2
    if lst[mid] > value:
        return binary_search(lst, value, low, high - 1)
    elif lst[mid] < value:
        return binary_search(lst, value, low + 1, high)
    else:
        return mid


def bseach(lst, value):
    low, high = 0, len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] < value:
            low = low + 1
        elif lst[mid] > value:
            high = high - 1
        else:
            return mid
    return -1


def bisectSearch(lst, x):
    from bisect import bisect_left
    i = bisect_left(lst, x)
    if i != len(lst) and lst[i] == x:
        return i


if __name__ == '__main__':
    lst = sorted([2, 5, 3, 8])
    print(bisectSearch(lst, 5))
    print(bseach(lst, 5))
