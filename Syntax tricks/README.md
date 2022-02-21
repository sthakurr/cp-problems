# Sorting with a key

Given an array of subarrays, we can sort it according to the first element and in case of a tie we can sort it wrt the second element too.

sorted(arr, key=(lambda i: (i[0], -i[0])))

