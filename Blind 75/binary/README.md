BITWISE Operators:

To obtain the binary rep of an integer: use bin(n)

Brian Kernighan’s Algorithm (to count the set bits in an integer's bit representation):
It says that the bit rep of (n-1) unsets the rightmost set bit of n and flips all the other values that are right to the rightmost set bit of n. So, if we do the bitwise & of n and (n-1), then we could unset the rightmost set bit! 

So,
[Link to the problem](https://leetcode.com/problems/number-of-1-bits/submissions/)
Solution using the Brian's Algo:
Class Solution(object):
    def hammingWeight(int n):
        count = 0
        while n > 0:
            n = n & (n-1)
            count += 1

    return count