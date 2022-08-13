## Interval Problems

** Sweepline ALgorithm: For all intervals inputs, this method should be the first intuition you come up with. **
It starts by sorting the list of intervals by the first element so all the operations can next be performed by comparing the end of an interval with the start of the next.

[Merge Intervals](https://www.interviewbit.com/problems/merge-intervals/)

[Sweep Line algo](https://leetcode.com/problems/maximum-sum-obtained-of-any-permutation/discuss/854206/JavaC%2B%2BPython-Sweep-Line)
sweep line algo: leetcode 253, 1589, 1109

## Array Math

** Kadane's Algorithm: It is a DP algo where we compare the dp[i-1]+A[i] with A[i] to assign dp[i] where dp[i] stores the answer till the subarray i. whenever a new subarray starts whose value of first element is greater than 

## Bucketing

Use a ** hashmap ** for finding the duplicate in an array

## Space Renewal *

Using the array elements as the pointers to mark if a number is visited or not.
[First Missing Integer](https://www.interviewbit.com/problems/first-missing-integer/)

NOTE: If we need to work with some numbers that can be treated as indices of the same array, use this technique!

```python
abs(A[i]) - 1 < len(A) and A[abs(A[i]) - 1] > 0:
    A[abs(A[i]) - 1] = #any value
```

## Simulation Array

Simulating/traversing the whole array: [Spiral Order Matrix II](https://www.interviewbit.com/problems/spiral-order-matrix-ii/)

## Sorting

Trick: Use flags for arrival and departure and then sort according to the arrivals so it is easier to keep a track of people coming and going

[Hotel Bookings Possible] = 
NOTE: sorted(arr, key=lambda i:[i[0], -i[1]]): use this key to sort according to the dates and then according to arrival/dep: prioritise arrival otherwise it'll treat as if the same day, a person first leaves and then the other arrives, which is not possible. So, first arrival then departure!

## Arrangement

Use a different comparator to sort according to that function! 
[Largest Number](https://www.interviewbit.com/problems/largest-number/)
