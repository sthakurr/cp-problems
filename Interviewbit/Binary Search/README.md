## Binary Search

Time complexity of O(logN) - most efficient searching algorithm in sorted arrays

**NOTE** - do not use library functions instead practice the explicit version of the algorithm

Recursive Implementation:
```python
def BinarySearch(arr, start, end, key):
    mid = (start+end)//2 ## or mid = start + (end - start)//2

    if start <= end:
        if arr[mid] == key:
            return mid
        else:
            if arr[mid] < key:
                return BinarySearch(arr, mid+1, end, key)
            else:
                return BinarySearch(arr, start, mid-1, key)
    else:
        return -1
```

Iterative involves a while loop with start and end values changing according to the conditions!

Where to use?
Find if a number is a square of any integer: To check if a number is a square of any integer, run a binary search from 1 to num and check if the square of mid is equal to num.
Dictionary: In the dictionary, all the words are arranged in lexicographical order, therefore, to find a particular word, we can simply binary search to find the alphabets without having to go through each word.

** Disadvantages **
Recursive approach uses stack space